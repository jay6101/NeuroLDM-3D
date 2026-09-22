"""
5-fold cross-validation training for the 3D CNN.

Scheme requested:
  * The TRAIN set is fixed (data_csvs/train.csv), optionally stratified-subsampled
    to `num_samples` real volumes and/or topped up with `num_synth_samples`
    synthetic volumes  ->  this is the SAME train set for every fold.
  * The VAL set and TEST set (data_csvs/val.csv + data_csvs/test.csv) are
    concatenated into a single "eval pool" and split into 5 stratified folds.
  * For each fold k (k = 1..5):
        - 1 fold  -> TEST   (~20% of the eval pool)
        - 4 folds -> VAL    (~80% of the eval pool, used for early stopping)
    Because the eval pool and split_seed are fixed, the 5 folds are identical
    across every run variant -> fair comparison.

  => 5 checkpoints + 5 result sets per variant.

Two model backends are supported (selected via hyperparams['model_key'] /
hyperparams['model_name']):
  * 3D CNN  (cnn_ben_3d)     : ReduceLROnPlateau, capped steps/epoch, volume
                               input [1, D, H, W]. Recipe from
                               3D_CNN/3D_supervised_HC_vs_TLE.
  * 2D CNN  (efficientNetV2) : CosineAnnealingWarmRestarts, full loader/epoch,
                               slices-as-channels input [D, H, W]. Recipe from
                               Diffusion_paper/classifier.
Both share BCEWithLogits + BalancedBatchSampler and the same dataset/folds.
"""

import os
import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from dataset import MRIDataset, BalancedBatchSampler
from utils import calculate_metrics, validate, save_fold_metrics, calculate_overall_metrics

CLASSES = ["right", "left", "HC"]


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_model_class(modelname):
    """Dynamically import and return the model class based on model name."""
    try:
        module = __import__(f'models.{modelname}', fromlist=['MRIClassifier'])
        return getattr(module, 'MRIClassifier')
    except ImportError:
        raise ImportError(f"Could not import model {modelname} from models package")
    except AttributeError:
        raise AttributeError(f"Model {modelname} does not contain MRIClassifier class")


def build_model(MRIClassifier, hyperparams):
    """Instantiate the model, passing only kwargs the backend accepts.

    `filters` is a cnn_ben_3d-specific argument; efficientNetV2 doesn't take it
    (its preset stores filters=None), so we omit it in that case.
    """
    kwargs = {'dropout_rate': hyperparams['dropout_rate']}
    if hyperparams.get('filters') is not None:
        kwargs['filters'] = hyperparams['filters']
    return MRIClassifier(**kwargs).to(hyperparams['device'])


def forward_logits(model, images):
    """Run a forward pass and return logits, unwrapping (logits, aux) tuples."""
    outputs = model(images)
    if isinstance(outputs, tuple):   # efficientNetV2 returns (logits, None)
        outputs = outputs[0]
    return outputs


def build_eval_folds(hyperparams):
    """Concatenate val + test CSVs and build the 5 stratified folds.

    Returns the combined eval dataframe and a list of (val_df, test_df) tuples,
    one per fold.
    """
    val_df = pd.read_csv(hyperparams['val_csv'])
    test_df = pd.read_csv(hyperparams['test_csv'])

    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    eval_df = eval_df.loc[eval_df["HC_vs_LTLE_vs_RTLE_string"].isin(CLASSES)].reset_index(drop=True)

    # IMPORTANT: in data_csvs, test.csv is a subset of val.csv (all 303 test
    # files also appear in val.csv). Concatenating them therefore creates
    # duplicate files; if we split without de-duplicating, the same subject can
    # land in both the val fold and the test fold of a split -> leakage. We
    # de-duplicate by file so the 5 folds are clean and disjoint.
    before = len(eval_df)
    eval_df = eval_df.drop_duplicates(subset='file', keep='first').reset_index(drop=True)
    dropped = before - len(eval_df)
    if dropped:
        print(f"[build_eval_folds] dropped {dropped} duplicate file(s) when combining "
              f"val+test -> {len(eval_df)} unique volumes")

    skf = StratifiedKFold(
        n_splits=hyperparams.get('n_folds', 5),
        shuffle=True,
        random_state=hyperparams['split_seed'],
    )

    folds = []
    y = eval_df["HC_vs_LTLE_vs_RTLE_string"]
    # StratifiedKFold yields (train_index, heldout_index):
    #   heldout_index (1 fold, ~20%) -> TEST
    #   train_index   (4 folds, ~80%) -> VAL
    for val_index, test_index in skf.split(eval_df, y):
        fold_val_df = eval_df.iloc[val_index].reset_index(drop=True)
        fold_test_df = eval_df.iloc[test_index].reset_index(drop=True)
        folds.append((fold_val_df, fold_test_df))

    return eval_df, folds


def _copy_source_files(run_dir, modelname):
    source_files = [
        'run.py',
        'config.py',
        'train.py',
        'utils.py',
        'dataset.py',
        f'models/{modelname}.py',
    ]
    for file in source_files:
        src_path = os.path.join(os.path.dirname(__file__), file)
        if os.path.exists(src_path):
            dst_path = os.path.join(run_dir, os.path.basename(file))
            try:
                shutil.copy2(src_path, dst_path)
            except shutil.SameFileError:
                pass


def train_model(hyperparams, modelname):
    MRIClassifier = get_model_class(modelname)
    set_random_seeds(hyperparams['random_seed'])

    run_dir = hyperparams['run_dir']
    os.makedirs(run_dir, exist_ok=True)

    # Save parameters
    hyperparams_to_save = hyperparams.copy()
    hyperparams_to_save['device'] = str(hyperparams_to_save['device'])
    hyperparams_to_save['saved_at'] = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(run_dir, 'parameters.json'), 'w') as f:
        json.dump(hyperparams_to_save, f, indent=4)

    _copy_source_files(run_dir, modelname)

    print(f"\n{'#' * 70}")
    print(f"# Variant: {hyperparams.get('variant_name', os.path.basename(run_dir))}")
    print(f"# Model: {modelname} ({hyperparams.get('model_key', '?')})  "
          f"input={'[1,D,H,W]' if hyperparams.get('volume_3d', True) else '[D,H,W]'}")
    print(f"# num_samples={hyperparams['num_samples']}  "
          f"num_synth_samples={hyperparams['num_synth_samples']}")
    print(f"# Using device: {hyperparams['device']}")
    print(f"{'#' * 70}")

    # Fixed train pool (same for every fold)
    full_train_df = pd.read_csv(hyperparams['train_csv'])
    full_train_df = full_train_df.loc[full_train_df["HC_vs_LTLE_vs_RTLE_string"].isin(CLASSES)]

    # Build the 5 (val, test) folds from the combined val+test pool
    eval_df, folds = build_eval_folds(hyperparams)
    # Optionally train only the first `max_folds` folds (handy for quick tests);
    # the split geometry (e.g. 80/20) is unchanged.
    max_folds = hyperparams.get('max_folds')
    if max_folds is not None:
        folds = folds[:max_folds]
        print(f"[max_folds] limiting to first {len(folds)} fold(s)")
    print(f"Eval pool (val+test) size: {len(eval_df)} | folds to train: {len(folds)}")

    all_folds_metrics = {'fold_metrics': []}
    fold_summary = []

    for fold_idx, (fold_val_df, fold_test_df) in enumerate(folds):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx + 1}/{len(folds)}")
        print(f"{'=' * 60}")

        fold_dir = os.path.join(run_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        # ----- datasets (train pool is shared; subsampling is deterministic) -----
        #   volume_3d selects the output shape for the chosen architecture:
        #     3D CNN  -> [1, D, H, W]   |   2D CNN -> [D, H, W]
        volume_3d = hyperparams.get('volume_3d', True)
        train_dataset = MRIDataset(
            full_train_df, hyperparams, train=True,
            num_samples=hyperparams['num_samples'],
            num_synth_samples=hyperparams['num_synth_samples'],
            volume_3d=volume_3d,
        )
        val_dataset = MRIDataset(fold_val_df, hyperparams, train=False, volume_3d=volume_3d)
        test_dataset = MRIDataset(fold_test_df, hyperparams, train=False, volume_3d=volume_3d)

        # Persist the exact data used for this fold.
        #   train.csv -> the actual samples trained on (real subsample and/or
        #   synthetic volumes) with a `source` column; for synthetic-only runs
        #   this contains only synthetic .pkl paths. val/test keep the full
        #   original (real) metadata columns.
        train_dataset.samples_dataframe().to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        fold_val_df.to_csv(os.path.join(fold_dir, 'val.csv'), index=False)
        fold_test_df.to_csv(os.path.join(fold_dir, 'test.csv'), index=False)

        n_real = sum(1 for p, _ in train_dataset.samples if '.pkl' not in p)
        n_synth = sum(1 for p, _ in train_dataset.samples if '.pkl' in p)
        print(f"Train: {len(train_dataset)} (real: {n_real}, synth: {n_synth})")
        print(f"Val:   {len(val_dataset)}")
        print(f"Test:  {len(test_dataset)}")

        # ----- dataloaders -----
        train_sampler = BalancedBatchSampler(train_dataset, hyperparams['batch_size'])
        train_loader = DataLoader(
            train_dataset, batch_sampler=train_sampler,
            num_workers=hyperparams['num_workers'], pin_memory=True, prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=hyperparams['batch_size'], shuffle=False,
            num_workers=hyperparams['num_workers'], pin_memory=True, prefetch_factor=2,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=hyperparams['batch_size'], shuffle=False,
            num_workers=hyperparams['num_workers'], pin_memory=True,
        )

        steps_per_epoch = max(
            1, len(train_dataset) // (hyperparams['batch_size'] * hyperparams.get('steps_multiplier', 1))
        )
        print(f"Steps per epoch: {steps_per_epoch}")

        # ----- model / loss / optim / scheduler (3D CNN reference setup) -----
        model = build_model(MRIClassifier, hyperparams)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay'],
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            patience=hyperparams['scheduler_patience'],
            factor=hyperparams['scheduler_factor'],
            min_lr=hyperparams.get('min_lr', 1e-8),
            threshold=hyperparams.get('scheduler_threshold', 1e-4),
        )

        fold_metrics = {
            'train_metrics': [], 'val_metrics': [], 'epochs': [],
            'best_val_loss': float('inf'), 'best_val_metrics': None,
            'test_metrics': None,
        }

        best_val_loss = float('inf')
        patience_counter = 0

        # ----- training loop -----
        for epoch in range(hyperparams['num_epochs']):
            model.train()
            running_loss = 0.0
            epoch_loss = 0.0
            all_train_labels, all_train_predictions, all_train_scores = [], [], []

            step_count = 0
            for i, (images, labels, _) in enumerate(train_loader):
                images = images.to(hyperparams['device'])
                labels = labels.float().to(hyperparams['device'])

                optimizer.zero_grad()
                outputs = forward_logits(model, images).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_loss += loss.item()
                scores = torch.sigmoid(outputs)
                predictions = (scores > 0.5).float()

                all_train_labels.extend(labels.cpu().detach().numpy())
                all_train_predictions.extend(predictions.cpu().detach().numpy())
                all_train_scores.extend(scores.cpu().detach().numpy())

                step_count += 1
                if i % 10 == 9:
                    print(f'Fold [{fold_idx + 1}] Epoch [{epoch + 1}/{hyperparams["num_epochs"]}], '
                          f'Step [{i + 1}/{steps_per_epoch}], Loss: {running_loss / 10:.4f}')
                    running_loss = 0.0

                if step_count >= steps_per_epoch:
                    break

            train_metrics = calculate_metrics(
                np.array(all_train_labels),
                np.array(all_train_predictions),
                np.array(all_train_scores),
            )
            train_metrics['loss'] = epoch_loss / step_count
            val_metrics, best_threshold = validate(
                model, val_loader, criterion, hyperparams, return_threshold=True
            )
            val_metrics['best_threshold'] = best_threshold

            print(f'Fold [{fold_idx + 1}] Epoch [{epoch + 1}/{hyperparams["num_epochs"]}]: '
                  f'Train AUC {train_metrics["auc_roc"]:.4f} | '
                  f'Val Loss {val_metrics["loss"]:.4f} AUC {val_metrics["auc_roc"]:.4f} '
                  f'Acc {val_metrics["accuracy"]:.4f}')

            scheduler.step(val_metrics['loss'])

            fold_metrics['train_metrics'].append(train_metrics)
            fold_metrics['val_metrics'].append(val_metrics)
            fold_metrics['epochs'].append(epoch)

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                fold_metrics['best_val_metrics'] = val_metrics.copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': val_metrics,
                }, os.path.join(fold_dir, 'best_model.pth'))
                print(f'  -> best model saved (val_loss {best_val_loss:.4f})')
            else:
                patience_counter += 1
                if patience_counter >= hyperparams['early_stopping_patience']:
                    print(f'Early stopping after {epoch + 1} epochs')
                    break

            fold_metrics['best_val_loss'] = best_val_loss
            save_fold_metrics(fold_metrics, fold_dir)

        # ----- test phase (best checkpoint) -----
        best_model = build_model(MRIClassifier, hyperparams)
        checkpoint = torch.load(os.path.join(fold_dir, 'best_model.pth'), weights_only=False)
        best_model.load_state_dict(checkpoint['model_state_dict'])

        threshold = (fold_metrics['best_val_metrics']['best_threshold']
                     if hyperparams['use_best_threshold'] else 0.5)
        test_metrics, individual_results = validate(
            best_model, test_loader, criterion, hyperparams,
            fixed_threshold=threshold, return_individual_results=True,
        )

        performance_labels = []
        for true_label, pred_label in zip(individual_results['labels'], individual_results['predictions']):
            if true_label == 1 and pred_label == 1:
                performance_labels.append('TP')
            elif true_label == 0 and pred_label == 1:
                performance_labels.append('FP')
            elif true_label == 1 and pred_label == 0:
                performance_labels.append('FN')
            else:
                performance_labels.append('TN')

        results_df = pd.DataFrame({
            'img_path': individual_results['paths'],
            'predicted_score': individual_results['scores'],
            'original_label': individual_results['labels'],
            'predicted_label': individual_results['predictions'],
            'performance_type': performance_labels,
        })
        results_df.to_csv(os.path.join(fold_dir, 'individual_test_performance.csv'), index=False)

        test_metrics['threshold_used'] = threshold
        fold_metrics['test_metrics'] = test_metrics
        save_fold_metrics(fold_metrics, fold_dir)
        all_folds_metrics['fold_metrics'].append(fold_metrics)

        print(f"\nFold {fold_idx + 1} TEST: "
              f"AUC {test_metrics['auc_roc']:.4f} | Acc {test_metrics['accuracy']:.4f} | "
              f"Sens {test_metrics['sensitivity']:.4f} | Spec {test_metrics['specificity']:.4f}")

        fold_summary.append({
            'fold': fold_idx + 1,
            'best_epoch': int(checkpoint['epoch']) + 1,
            'test_metrics': {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in test_metrics.items()},
        })

    # ----- aggregate over the 5 test folds -----
    overall_results = calculate_overall_metrics(all_folds_metrics)

    # all_folds_metrics.json -> same format as the original new_runs:
    #   {"overall_results": {metric: {"mean": .., "std": ..}}}
    with open(os.path.join(run_dir, 'all_folds_metrics.json'), 'w') as f:
        json.dump({'overall_results': overall_results}, f, indent=4)

    # run_summary.json -> richer detail (config + per-fold test metrics) so we
    # don't lose information that isn't in the reference format.
    summary = {
        'variant_name': hyperparams.get('variant_name'),
        'num_samples': hyperparams['num_samples'],
        'num_synth_samples': hyperparams['num_synth_samples'],
        'n_folds': hyperparams.get('n_folds', 5),
        'folds_trained': len(folds),
        'split_seed': hyperparams['split_seed'],
        'per_fold': fold_summary,
        'overall_results': overall_results,
    }
    with open(os.path.join(run_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nOverall {len(folds)}-Fold CV results for "
          f"{hyperparams.get('variant_name', '')}:")
    print("-" * 50)
    for metric, values in overall_results.items():
        print(f"{metric.upper():12s}: {values['mean']:.4f} +/- {values['std']:.4f}")
    print("-" * 50)

    return summary
