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


def resolve_resume_checkpoint(output, load_path, gpus):
    from veomni.checkpoint.layout import checkpoint_is_complete, read_manifest

    if not load_path:
        return None
    if load_path == "auto":
        candidates = sorted(
            (p for p in (output / "checkpoints").glob("global_step_*") if p.name[12:].isdigit()),
            key=lambda p: int(p.name[12:]),
            reverse=True,
        )
        checkpoint = next((p for p in candidates if checkpoint_is_complete(str(p))), None)
    else:
        checkpoint = Path(load_path).resolve()
    if checkpoint is None or not checkpoint_is_complete(str(checkpoint)):
        raise ValueError("Resume requested but no complete checkpoint exists; use a new run name for fresh training")
    if read_manifest(str(checkpoint)).get("world_size") != gpus:
        raise ValueError("Resume requires the same GPU count as the saved rank-local data cursors")
    for rank in range(gpus):
        for directory in ("loader", "extra_state"):
            if not (checkpoint / directory / f"rank_{rank}.pt").is_file():
                raise ValueError(f"Resume checkpoint is missing {directory}/rank_{rank}.pt")
    return str(checkpoint)


def prepare_run(args):
    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    online = manifest.get("tokenization", "offline") == "online"
    if online:
        if not manifest.get("data_ready") or not manifest.get("raw_samples"):
            raise ValueError("Online data preparation has not finished")
        from veomni.data.maple import raw_shard_identity

        if raw_shard_identity(root, manifest["requested_shards"]) != manifest["raw_shards"]:
            raise ValueError("Raw dataset changed since preparation; prepare a new root")
    elif not manifest.get("train_samples"):
        raise ValueError("Data preparation has not finished; no verified train_samples in manifest")
    if manifest.get("model_revision", "").startswith("local-"):
        from veomni.models.transformers.maple.provenance import local_model_revision

        if local_model_revision(Path(manifest["model_path"]).resolve()) != manifest["model_revision"]:
            raise ValueError("Local model assets changed since preparation; prepare a new root")
    expected = {f"train-{i:05d}-of-00120.parquet" for i in range(manifest["requested_shards"])}
    for split in () if online else ("train", "validation"):
        files = {p.relative_to(root / split).as_posix() for p in (root / split).rglob("*") if p.is_file()}
        if files != expected:
            raise ValueError(f"Prepared {split} shards do not match the manifest; finish preparation first")
    config = yaml.safe_load(args.config.read_text())
    if config["data"]["max_seq_len"] != manifest["max_length"]:
        raise ValueError("data.max_seq_len must match the prepared manifest's max_length")
    batch_multiple = args.gpus * config["train"]["micro_batch_size"]
    if config["train"]["global_batch_size"] % batch_multiple:
        raise ValueError(f"global_batch_size must be divisible by GPUs * micro_batch_size ({batch_multiple})")
    output = root / args.run_name
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        raise ValueError("--run-name must be a single directory name")
    config["train"]["checkpoint"]["load_path"] = resolve_resume_checkpoint(
        output, config["train"]["checkpoint"].get("load_path"), args.gpus
    )
    if not config["train"]["checkpoint"].get("load_path") and any(
        (output / name).exists() for name in ("checkpoints", "training_clock.json", "metrics.jsonl")
    ):
        raise ValueError("Fresh training requires a new --run-name; this directory already contains training state")
    if not args.prepare_only and not config["train"]["checkpoint"].get("load_path"):
        keep = int(os.environ.get("MAPLE_KEEP_CHECKPOINTS", "1"))
        if keep < 1:
            raise ValueError("MAPLE_KEEP_CHECKPOINTS must be positive")
        fsdp = config["model"].get("accelerator", {}).get("fsdp_config", {})
        # ModelRuntime keeps FP32 master weights when mixed precision is on;
        # DCP saves those masters, not the BF16 tensors used during forward.
        weight_gib = 80 if fsdp.get("mixed_precision", {}).get("enable", True) else 40
        # Four weight copies conservatively cover both AdamW and AnyPrecision
        # states (including its compensation buffer), plus the weights.
        checkpoint_gib = weight_gib * (4 if config["train"]["checkpoint"].get("save_optimizer", True) else 1)
        required_gib = checkpoint_gib * (keep + 1)
        if online and config["data"].get("datasets_type") == "mapping":
            # HF's native map-style reader builds an Arrow cache of raw text.
            # Leave headroom for decoded columns as well as checkpoint rotation.
            cache = root / "dataset_cache"
            cached = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
            remaining = max(0, int(manifest["raw_uncompressed_bytes"] * 1.5) - cached)
            required_gib += (remaining + 1024**3 - 1) // 1024**3
        if shutil.disk_usage(root).free < required_gib * 1024**3:
            raise ValueError(f"Need {required_gib} GiB free for Maple checkpoints and the selected data cache")
    config["model"].update(
        model_path=manifest.get("model_path", str(root / "model")),
        config_path=str(root / "config"),
        tokenizer_path=str(root / "tokenizer"),
    )
    if online:
        if config["data"].get("train_size", 0) <= 0:
            raise ValueError("Online tokenization requires a positive data.train_size token budget")
        config["data"].update(
            train_path=str(root / "raw" / "data"), data_type="conversation", text_keys="conversations"
        )
    else:
        config["data"].update(
            train_path=str(root / "train"), train_size=manifest["train_tokens"], data_type="pretokenized"
        )
    output.mkdir(exist_ok=True)
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
        # Keep the raw Arrow cache off the small root filesystem. This is set
        # before the training subprocess imports datasets.
        os.environ["HF_DATASETS_CACHE"] = str(args.root.resolve() / "dataset_cache")
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
