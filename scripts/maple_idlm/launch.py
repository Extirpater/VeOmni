"""Resolve a Maple run config and launch torchrun."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def prepare_run(args):
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("train_samples"):
        raise ValueError("Data preparation has not finished; no verified train_samples in manifest")
    config = yaml.safe_load(args.config.read_text())
    batch_multiple = args.gpus * config["train"]["micro_batch_size"]
    if config["train"]["global_batch_size"] % batch_multiple:
        raise ValueError(f"global_batch_size must be divisible by GPUs * micro_batch_size ({batch_multiple})")
    output = root / args.run_name
    output.mkdir(exist_ok=True)
    if not config["train"]["checkpoint"].get("load_path") and any(
        (output / name).exists() for name in ("checkpoints", "training_clock.json", "metrics.jsonl")
    ):
        raise ValueError("Fresh training requires a new --run-name; this directory already contains training state")
    if not args.prepare_only and not config["train"]["checkpoint"].get("load_path"):
        if shutil.disk_usage(root).free < 320 * 1024**3:
            raise ValueError("Need 320 GiB free to rotate full Maple checkpoints")
    config["model"].update(
        model_path=str(root / "model"), config_path=str(root / "config"), tokenizer_path=str(root / "tokenizer")
    )
    config["data"].update(train_path=str(root / "train"), train_size=manifest["train_tokens"])
    shutil.copyfile(root / "manifest.json", output / "data_manifest.json")
    config["train"]["checkpoint"]["output_dir"] = str(output)
    if args.steps is not None:
        config["train"]["max_steps"] = args.steps
    target = output / "resolved.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/maple_idlm"))
    parser.add_argument("--config", type=Path, default=Path("configs/text/maple_idlm.yaml"))
    parser.add_argument("--run-name", default=time.strftime("run-fresh-%Y%m%dT%H%M%S", time.gmtime()))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--fresh-window", action="store_true", help="Resume with a fresh wall-time limit.")
    args = parser.parse_args()
    if args.gpus < 1 or args.hours <= 0 or (args.steps is not None and args.steps < 1):
        parser.error("--gpus, --hours and --steps must be positive")
    if not all((args.root / name).is_file() for name in ("manifest.json", "config/config.json")):
        parser.error("Run scripts/maple_idlm/prepare_data.py first")
    # Hold one lock across preparation and training, shared by every run in this root.
    with (args.root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = prepare_run(args)
        print(f"Config: {target}\nLog: {target.parent / 'training.log'}", flush=True)
        if args.prepare_only:
            return
        clock = target.parent / "training_clock.json"
        if args.fresh_window and clock.exists():
            clock.rename(clock.with_name(f"training_clock.{time.time_ns()}.json"))
        os.environ["MAPLE_RUN_SECONDS"] = str(args.hours * 3600)
        os.environ.setdefault("OMP_NUM_THREADS", str(max(1, min(16, (os.cpu_count() or 1) // args.gpus))))
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.gpus}",
            "tasks/train_maple_idlm.py",
            str(target),
        ]
        with (target.parent / "training.log").open("a") as log:
            raise SystemExit(subprocess.call(command, stdout=log, stderr=subprocess.STDOUT))


if __name__ == "__main__":
    main()
