"""
Multi-GPU queue runner for all 3D-CNN CV variants.

Runs up to `jobs_per_gpu` variants concurrently on each GPU. When a job
finishes, the next queued variant is started on that same GPU.

Each variant is launched as its own subprocess (isolated CUDA context) via
run.py, with CUDA_VISIBLE_DEVICES pinning it to a physical GPU.

Examples
--------
# Default: 3D CNN, 2 jobs/GPU on GPUs 0 and 1  -> 4 concurrent variants
conda activate jay2
cd .../new_runs_3dcnn_cv
python run_queue.py

# Train the 2D EfficientNetV2 instead (writes to runs_2d/)
python run_queue.py --model 2d

# 1 job per GPU (safer on memory; recommended for the heavier 2D model)
python run_queue.py --jobs-per-gpu 1

# Resume: skip variants that already have all_folds_metrics.json
python run_queue.py --skip-existing

# Dry-run the schedule
python run_queue.py --dry-run

# Run in background with nohup
nohup python run_queue.py --skip-existing > queue_master.log 2>&1 &
tail -f queue_master.log

# Watch per-variant verbose logs (written to runs/logs/)
python watch_logs.py --list                  # list all log files
python watch_logs.py --follow real_500       # tail one variant live
python watch_logs.py --follow-running        # tail all currently running jobs
python watch_logs.py --follow-all            # tail every log file at once
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import config

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PYTHON = "/home/jsawant/.conda/envs/jay2/bin/python"


def select_variants(groups=None, variants=None):
    selected = config.VARIANTS
    if variants:
        wanted = {v.strip() for v in variants.split(",")}
        selected = [v for v in selected if v['name'] in wanted]
        missing = wanted - {v['name'] for v in selected}
        if missing:
            raise SystemExit(f"Unknown variant(s): {sorted(missing)}")
    elif groups:
        wanted = {g.strip() for g in groups.split(",")}
        selected = [v for v in selected if v['group'] in wanted]
        missing = wanted - {v['group'] for v in selected}
        if missing:
            raise SystemExit(f"Unknown group(s): {sorted(missing)}")
    return selected


def is_done(variant, runs_dir):
    marker = os.path.join(runs_dir, variant['group'], variant['name'], 'all_folds_metrics.json')
    return os.path.exists(marker)


def build_queue(variants, runs_dir, skip_existing):
    queue = []
    skipped = []
    for v in variants:
        if skip_existing and is_done(v, runs_dir):
            skipped.append(v)
        else:
            queue.append(v)
    return queue, skipped


def variant_log_path(runs_dir, variant):
    log_dir = os.path.join(runs_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f"{variant['group']}_{variant['name']}.log")


def launch_variant(python, variant, gpu_id, runs_dir, extra_args):
    """Start one variant as a subprocess pinned to `gpu_id`."""
    log_path = variant_log_path(runs_dir, variant)
    log_f = open(log_path, "a", buffering=1)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        python, os.path.join(THIS_DIR, "run.py"),
        "--variants", variant["name"],
        "--device", "cuda:0",          # logical cuda:0 == physical gpu_id
        "--runs-dir", runs_dir,
    ]
    cmd.extend(extra_args)

    header = (
        f"\n{'='*70}\n"
        f"[{datetime.now().isoformat(timespec='seconds')}] "
        f"START {variant['group']}/{variant['name']} on GPU {gpu_id}\n"
        f"CMD: {' '.join(cmd)}\n"
        f"{'='*70}\n"
    )
    log_f.write(header)
    log_f.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=THIS_DIR,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    return proc, log_f, log_path


def poll_slots(slots):
    """Check running jobs; return list of freed gpu_ids."""
    freed = []
    for gpu_id, entries in list(slots.items()):
        still_running = []
        for entry in entries:
            ret = entry["proc"].poll()
            if ret is None:
                still_running.append(entry)
            else:
                entry["log_f"].write(
                    f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                    f"END {entry['variant']['group']}/{entry['variant']['name']} "
                    f"on GPU {gpu_id} exit_code={ret}\n"
                )
                entry["log_f"].close()
                freed.append(gpu_id)
                entry["exit_code"] = ret
                entry["end_time"] = datetime.now().isoformat(timespec="seconds")
        slots[gpu_id] = still_running
    return freed


def count_running(slots):
    return sum(len(v) for v in slots.values())


def write_status_file(runs_dir, slots, queue, completed, failed, gpu_ids, jobs_per_gpu):
    """Live status file for watch_logs.py and external monitoring."""
    running = []
    for gpu_id in sorted(slots):
        for entry in slots[gpu_id]:
            v = entry["variant"]
            running.append({
                "variant": v["name"],
                "group": v["group"],
                "gpu": gpu_id,
                "pid": entry["proc"].pid,
                "log": entry["log_path"],
                "start_time": entry["start_time"],
            })

    status = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": gpu_ids,
        "jobs_per_gpu": jobs_per_gpu,
        "running": running,
        "queued": [v["name"] for v in queue],
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed": [r["variant"] for r in completed],
        "failed": [r["variant"] for r in failed],
        "logs_dir": os.path.join(runs_dir, "logs"),
    }
    path = os.path.join(runs_dir, "queue_status.json")
    with open(path, "w") as f:
        json.dump(status, f, indent=2)
    return path


def print_status(slots, queue, completed, failed):
    running_desc = []
    for gpu_id in sorted(slots):
        for e in slots[gpu_id]:
            v = e["variant"]
            running_desc.append(f"GPU{gpu_id}:{v['name']}")
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        f"running={count_running(slots)} "
        f"queued={len(queue)} "
        f"done={len(completed)} "
        f"failed={len(failed)}"
        + (f"  |  {', '.join(running_desc)}" if running_desc else "")
    )


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU queue runner")
    parser.add_argument("--gpus", default="0,1", help="comma-separated physical GPU ids")
    parser.add_argument("--jobs-per-gpu", type=int, default=2,
                        help="max concurrent variants per GPU (default: 2)")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="python executable")
    parser.add_argument("--model", default=config.DEFAULT_MODEL, choices=config.model_choices(),
                        help="architecture: 3d (cnn_ben_3d) or 2d (efficientNetV2)")
    parser.add_argument("--runs-dir", default=None,
                        help="output root (default: runs/ for 3d, runs_2d/ for 2d)")
    parser.add_argument("--groups", default=None, help="comma-separated groups to run")
    parser.add_argument("--variants", default=None, help="comma-separated variant names")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip variants with all_folds_metrics.json")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="passed through to run.py (lower if CPU/NFS saturated)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15.0,
                        help="status poll interval")
    args = parser.parse_args()

    if args.runs_dir is None:
        args.runs_dir = config.default_runs_dir(args.model)

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    max_concurrent = len(gpu_ids) * args.jobs_per_gpu

    variants = select_variants(args.groups, args.variants)
    queue, skipped = build_queue(variants, args.runs_dir, args.skip_existing)

    extra_args = ["--model", args.model]
    if args.num_workers is not None:
        extra_args.extend(["--num-workers", str(args.num_workers)])

    print(f"Python     : {args.python}")
    print(f"Model      : {args.model} ({config.MODELS[args.model]['model_name']})")
    print(f"GPUs       : {gpu_ids}  ({args.jobs_per_gpu} jobs/GPU -> {max_concurrent} concurrent)")
    print(f"Runs dir   : {args.runs_dir}")
    print(f"Variants   : {len(variants)} total | {len(skipped)} skip | {len(queue)} to run")
    print(f"Logs       : {os.path.join(args.runs_dir, 'logs')}/")

    if args.dry_run:
        print("\n[dry-run] launch order:")
        for i, v in enumerate(queue, 1):
            gpu = gpu_ids[(i - 1) % len(gpu_ids)]
            print(f"  {i:2d}. {v['group']}/{v['name']}  (example GPU {gpu})")
        return

    if not queue:
        print("Nothing to run.")
        return

    if not os.path.exists(args.python):
        raise SystemExit(f"Python not found: {args.python}")

    # gpu_id -> list of running job dicts
    slots = {g: [] for g in gpu_ids}
    completed = []
    failed = []
    history = []

    def gpu_has_capacity(gpu_id):
        return len(slots[gpu_id]) < args.jobs_per_gpu

    def launch_on_gpu(gpu_id):
        if not queue or not gpu_has_capacity(gpu_id):
            return False
        variant = queue.pop(0)
        proc, log_f, log_path = launch_variant(
            args.python, variant, gpu_id, args.runs_dir, extra_args
        )
        entry = {
            "variant": variant,
            "proc": proc,
            "log_f": log_f,
            "log_path": log_path,
            "gpu_id": gpu_id,
            "start_time": datetime.now().isoformat(timespec="seconds"),
        }
        slots[gpu_id].append(entry)
        print(f"  -> launched {variant['group']}/{variant['name']} on GPU {gpu_id}  "
              f"(pid {proc.pid}, log {log_path})")
        return True

    # Fill initial slots
    print("\nFilling initial slots...")
    while queue and count_running(slots) < max_concurrent:
        launched_any = False
        for gpu_id in gpu_ids:
            while gpu_has_capacity(gpu_id) and queue:
                if launch_on_gpu(gpu_id):
                    launched_any = True
                if count_running(slots) >= max_concurrent:
                    break
        if not launched_any:
            break

    print_status(slots, queue, completed, failed)
    status_path = write_status_file(
        args.runs_dir, slots, queue, completed, failed, gpu_ids, args.jobs_per_gpu
    )
    print(f"Status file: {status_path}")

    # Main loop
    try:
        while count_running(slots) > 0 or queue:
            time.sleep(args.poll_seconds)

            # Collect finished jobs
            for gpu_id in gpu_ids:
                finished = []
                remaining = []
                for entry in slots[gpu_id]:
                    ret = entry["proc"].poll()
                    if ret is None:
                        remaining.append(entry)
                    else:
                        finished.append((entry, ret))
                slots[gpu_id] = remaining

                for entry, ret in finished:
                    v = entry["variant"]
                    entry["log_f"].write(
                        f"\n[{datetime.now().isoformat(timespec='seconds')}] "
                        f"END {v['group']}/{v['name']} on GPU {gpu_id} exit_code={ret}\n"
                    )
                    entry["log_f"].close()
                    record = {
                        "variant": v["name"],
                        "group": v["group"],
                        "gpu": gpu_id,
                        "exit_code": ret,
                        "start_time": entry["start_time"],
                        "end_time": datetime.now().isoformat(timespec="seconds"),
                        "log": entry["log_path"],
                    }
                    history.append(record)
                    if ret == 0:
                        completed.append(record)
                        print(f"  ✓ finished {v['group']}/{v['name']} on GPU {gpu_id}")
                    else:
                        failed.append(record)
                        print(f"  ✗ FAILED  {v['group']}/{v['name']} on GPU {gpu_id} (exit {ret})")

                    # Refill this GPU
                    while gpu_has_capacity(gpu_id) and queue:
                        launch_on_gpu(gpu_id)
                        if count_running(slots) >= max_concurrent:
                            break

            if count_running(slots) > 0 or queue:
                print_status(slots, queue, completed, failed)
                write_status_file(
                    args.runs_dir, slots, queue, completed, failed, gpu_ids, args.jobs_per_gpu
                )

    except KeyboardInterrupt:
        print("\n[interrupt] terminating running jobs...")
        for gpu_id in gpu_ids:
            for entry in slots[gpu_id]:
                entry["proc"].terminate()
        raise

    # Write master summary
    summary_path = os.path.join(args.runs_dir, "queue_summary.json")
    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "gpus": gpu_ids,
        "jobs_per_gpu": args.jobs_per_gpu,
        "skipped": [v["name"] for v in skipped],
        "completed": [r["variant"] for r in completed],
        "failed": [r["variant"] for r in failed],
        "history": history,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Queue finished.")
    print(f"  completed : {len(completed)}")
    print(f"  failed    : {len(failed)}")
    print(f"  skipped   : {len(skipped)}")
    print(f"  summary   : {summary_path}")
    print(f"{'='*60}")

    if failed:
        print("\nFailed variants:")
        for r in failed:
            print(f"  - {r['group']}/{r['variant']}  (log: {r['log']})")
        sys.exit(1)


if __name__ == "__main__":
    main()
