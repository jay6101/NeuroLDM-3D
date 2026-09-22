"""
Metrics / validation helpers.

Copied (essentially unchanged) from
3D_CNN/3D_supervised_HC_vs_TLE/utils.py so the evaluation behaviour matches
the 3D CNN reference. `validate` accepts either raw logits (a single tensor,
cnn_ben_3d) or a (logits, aux) tuple (efficientNetV2) and unwraps the latter.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    average_precision_score,
    accuracy_score,
)
import os
import json


def calculate_metrics(labels, predictions, scores, return_threshold=False):
    """Calculate various classification metrics."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[optimal_idx]

    if predictions is None:
        predictions = (scores >= best_threshold).astype(float)

    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'auc_roc': roc_auc_score(labels, scores),
        'auc_pr': average_precision_score(labels, scores),
        'ppv': precision_score(labels, predictions, zero_division=0),
        'sensitivity': recall_score(labels, predictions, zero_division=0),
        'specificity': recall_score(labels, predictions, pos_label=0, zero_division=0),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1_score': f1_score(labels, predictions, zero_division=0),
        'best_threshold': best_threshold,
    }

    if return_threshold:
        return metrics, best_threshold
    return metrics


def validate(model, dataloader, criterion, hyperparams, fixed_threshold=None,
             return_threshold=False, return_individual_results=False):
    """Validate the model on the provided dataloader."""
    model.eval()
    val_loss = 0
    all_labels = []
    all_scores = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(hyperparams['device'])
            labels = labels.float().to(hyperparams['device'])

            outputs = model(images)
            if isinstance(outputs, tuple):   # 2D EfficientNetV2 returns (logits, None)
                outputs = outputs[0]
            outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            scores = torch.sigmoid(outputs)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_paths.extend(paths)

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    threshold = fixed_threshold if fixed_threshold is not None else 0.5
    all_predictions = (all_scores >= threshold).astype(float)

    metrics = calculate_metrics(all_labels, all_predictions, all_scores)
    metrics['loss'] = val_loss / len(dataloader)

    if return_individual_results:
        individual_results = {
            'paths': all_paths,
            'scores': all_scores,
            'labels': all_labels,
            'predictions': all_predictions,
        }
        return metrics, individual_results

    if return_threshold:
        return metrics, metrics['best_threshold']

    return metrics


def save_fold_metrics(metrics, fold_dir):
    """Save metrics to JSON, converting numpy / torch types to native Python."""
    def convert_to_python_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        return obj

    metrics_to_save = convert_to_python_types(metrics)

    with open(os.path.join(fold_dir, 'fold_metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=4)


def calculate_overall_metrics(all_folds_metrics):
    """Calculate mean and std for all test metrics across folds."""
    metrics_to_calculate = [
        'accuracy', 'auc_roc', 'auc_pr', 'ppv',
        'sensitivity', 'specificity', 'precision',
        'recall', 'f1_score',
    ]

    overall_results = {}
    for metric in metrics_to_calculate:
        values = [float(fold['test_metrics'][metric])
                  for fold in all_folds_metrics['fold_metrics']]
        overall_results[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
        }

    return overall_results
