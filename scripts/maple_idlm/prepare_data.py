"""Prepare OpenThoughts3 assets for online or offline Maple tokenization."""

import argparse
import fcntl
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from veomni.data.maple import conversation_split, encode_conversation
from veomni.models.transformers.maple.provenance import local_model_revision


MODEL_ID = "deepgrove/maple-preview"
MODEL_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
DATA_ID = "open-thoughts/OpenThoughts3-1.2M"
DATA_REVISION = "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"
PREPARATION_VERSION = 1
_TOKENIZER = None


def validate_preparation_root(root, manifest):
    """Never mix tokenizations or shard selections in a directory dataset."""
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("tokenization", "offline") != manifest.get("tokenization", "offline"):
            raise ValueError("Preparation changed tokenization mode; choose a new --root")
        for key in ("model_revision", "dataset_revision", "max_length", "requested_shards", "ternary_scheme"):
            if previous.get(key) != manifest.get(key):
                raise ValueError(f"Preparation changed {key}; choose a new --root to preserve existing data")
    expected = {f"train-{i:05d}-of-00120.parquet" for i in range(manifest["requested_shards"])}
    for split in ("train", "validation"):
        existing = {
            path.relative_to(root / split).as_posix()
            for path in (root / split).rglob("*")
            if path.is_file() and path.suffix != ".tmp"
        }
        if existing and not manifest_path.exists():
            raise ValueError(f"Existing {split} data has no manifest; choose a new --root")
        if existing - expected:
            raise ValueError(f"Unexpected shards in {split}; choose a new --root")


def init_worker(tokenizer_path):
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, fix_mistral_regex=False)


def prepare_shard(job):
    root, file, max_length, model_revision = job
    marker = root / "preparation" / (Path(file).stem + ".json")
    signature = dict(version=PREPARATION_VERSION, max_length=max_length, model=model_revision, data=DATA_REVISION)
    if marker.exists():
        cached = json.loads(marker.read_text())
        if cached["signature"] == signature and all(
            (root / split / Path(file).name).exists() for split in ("train", "validation")
        ):
            return file, cached["stats"]
    schema = pa.schema([("input_ids", pa.list_(pa.int32())), ("labels", pa.list_(pa.int32()))])
    stats = {"train_samples": 0, "train_tokens": 0, "validation_samples": 0, "skipped": 0}
    outputs = {}
    for split in ("train", "validation"):
        directory = root / split
        directory.mkdir(exist_ok=True)
        outputs[split] = pq.ParquetWriter(directory / (Path(file).name + ".tmp"), schema, compression="zstd")
    try:
        for batch in pq.ParquetFile(root / "raw" / file).iter_batches(batch_size=128):
            rows = {"train": [], "validation": []}
            for row in batch.to_pylist():
                conversation = row.get("conversations", row.get("messages"))
                if not conversation:
                    raise ValueError(f"No conversation field in {file}")
                encoded = encode_conversation(conversation, _TOKENIZER, max_length)
                if encoded is None:
                    stats["skipped"] += 1
                    continue
                split = conversation_split(conversation)
                rows[split].append(encoded)
                stats[f"{split}_samples"] += 1
                if split == "train":
                    stats["train_tokens"] += len(encoded["input_ids"])
            for split, records in rows.items():
                if records:
                    outputs[split].write_table(pa.Table.from_pylist(records, schema=schema))
    finally:
        for writer in outputs.values():
            writer.close()
    for split in outputs:
        source = root / split / (Path(file).name + ".tmp")
        source.replace(source.with_suffix(""))
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(json.dumps(dict(signature=signature, stats=stats)))
    return file, stats


def prepare_data(args):
    model_dir = args.model_path.resolve() if args.model_path else args.root / "model"
    model_revision = local_model_revision(model_dir) if args.model_path else MODEL_REVISION
    manifest = {
        "model": str(model_dir) if args.model_path else MODEL_ID,
        "model_path": str(model_dir.resolve()),
        "model_revision": model_revision,
        "ternary_scheme": args.ternary_scheme,
        "dataset": DATA_ID,
        "dataset_revision": DATA_REVISION,
        "paper_corpus": False,
        "max_length": args.max_length,
        "requested_shards": args.shards,
        "tokenization": args.tokenization,
    }
    validate_preparation_root(args.root, manifest)
    # Tokenization needs only metadata; let a concurrent weight download finish
    # while the CPU workers prepare data. The final manifest still requires both.
    patterns = ["*.json", "*.jinja", "*.txt", "LICENSE", "README.md"]
    if not args.model_path:
        snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir, allow_patterns=patterns, max_workers=4
        )
    # The original config omits model_type, which triggers v5's Mistral heuristic.
    # Maple uses Qwen2 tokenization; preserve the pinned pretokenizer unchanged.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False, fix_mistral_regex=False)
    # Reserve a previously unused padded-vocabulary row without resizing weights.
    mask_name = "<|idlm_mask|>"
    tokenizer.add_special_tokens({"mask_token": mask_name})
    model_config = json.loads((model_dir / "config.json").read_text())
    if len(tokenizer) > model_config["vocab_size"]:
        raise ValueError("Maple tokenizer has no unused vocabulary row for MASK")
    tokenizer.save_pretrained(args.root / "tokenizer")
    model_config.update(model_type="maple", mask_token_id=tokenizer.mask_token_id, ternary_scheme=args.ternary_scheme)
    model_config.pop("auto_map", None)
    config_dir = args.root / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    manifest["mask_token_id"] = tokenizer.mask_token_id
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = [f"data/train-{i:05d}-of-00120.parquet" for i in range(args.shards)]
    snapshot_download(
        DATA_ID,
        repo_type="dataset",
        revision=DATA_REVISION,
        local_dir=args.root / "raw",
        allow_patterns=files + ["README.md"],
        max_workers=4,
    )
    print("Pinned tokenizer and dataset downloaded", flush=True)
    if args.tokenization == "online":
        from veomni.data.maple import raw_shard_identity

        manifest["raw_shards"] = raw_shard_identity(args.root, args.shards)
        metadata = [pq.ParquetFile(args.root / "raw" / file).metadata for file in files]
        manifest["raw_samples"] = sum(item.num_rows for item in metadata)
        manifest["raw_uncompressed_bytes"] = sum(
            item.row_group(index).total_byte_size for item in metadata for index in range(item.num_row_groups)
        )
    else:
        stats = {"train_samples": 0, "train_tokens": 0, "validation_samples": 0, "skipped": 0}
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker,
            initargs=(str(args.root / "tokenizer"),),
        ) as pool:
            for file, counts in pool.map(
                prepare_shard, [(args.root, file, args.max_length, model_revision) for file in files]
            ):
                for key, value in counts.items():
                    stats[key] += value
                print(json.dumps({"prepared": file, **stats}), flush=True)
        manifest.update(stats)
    if args.model_path:
        if local_model_revision(model_dir) != model_revision:
            raise ValueError("Local checkpoint changed during preparation; choose an immutable model directory")
    else:
        snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir, allow_patterns=["*.safetensors"], max_workers=4
        )
    manifest["data_ready"] = True
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/maple_idlm"))
    parser.add_argument(
        "--model-path", type=Path, help="Reuse a local indexed Maple checkpoint without downloading weights"
    )
    parser.add_argument("--ternary-scheme", choices=("group_absmax", "row_twn"), default="group_absmax")
    parser.add_argument(
        "--tokenization", choices=("online", "offline"), default="offline", help="Online tokenizes in training workers"
    )
    parser.add_argument("--shards", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.shards <= 120:
        parser.error("--shards must be between 1 and 120")
    if args.max_length < 2 or args.workers < 1:
        parser.error("--max-length must be at least 2 and --workers must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    # The launcher uses the same lock: never replace data beneath a live run.
    with (args.root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_data(args)


if __name__ == "__main__":
    main()
