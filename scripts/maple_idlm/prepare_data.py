"""Download pinned Maple/OpenThoughts3 snapshots and prepare response-only tokens."""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


MODEL_ID = "deepgrove/maple-preview"
MODEL_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
DATA_ID = "open-thoughts/OpenThoughts3-1.2M"
DATA_REVISION = "61bcf9d4eb38b30295efc2021227a63cc5bb34c8"
PREPARATION_VERSION = 1
_TOKENIZER = None


def encode_conversation(conversation, tokenizer, max_length):
    messages = [
        {
            "role": {"human": "user", "gpt": "assistant"}.get(m.get("from"), m.get("role")),
            "content": m.get("value", m.get("content", "")),
        }
        for m in conversation
    ]
    ids, labels = [], []
    # Match Maple's native chat template, retaining reasoning traces verbatim.
    for index, message in enumerate(messages):
        prefix = (
            tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=False)
            if index
            else ""
        )
        rendered = tokenizer.apply_chat_template(messages[: index + 1], tokenize=False, add_generation_prompt=False)
        if not rendered.startswith(prefix):
            raise ValueError("Tokenizer template rewrites previous turns; cannot derive a safe assistant mask")
        tokens = tokenizer.encode(rendered[len(prefix) :], add_special_tokens=False)
        ids.extend(tokens)
        labels.extend(tokens if message["role"] == "assistant" else [-100] * len(tokens))
    # Keep the first max_length tokens; the unused suffix is not another sample.
    ids, labels = ids[:max_length], labels[:max_length]
    if len(ids) < 2 or sum(label != -100 for label in labels[1:]) < 2:
        return None
    return {"input_ids": ids, "labels": labels}


def init_worker(tokenizer_path):
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, fix_mistral_regex=False)


def prepare_shard(job):
    root, file, max_length = job
    marker = root / "preparation" / (Path(file).stem + ".json")
    signature = dict(version=PREPARATION_VERSION, max_length=max_length, model=MODEL_REVISION, data=DATA_REVISION)
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
                # Hash the prompt so repeated prompts cannot cross the split.
                prompt = conversation[0].get("value", conversation[0].get("content", ""))
                digest = hashlib.sha256(prompt.encode()).digest()
                split = "validation" if int.from_bytes(digest[:4], "big") % 100 == 0 else "train"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/maple_idlm"))
    parser.add_argument("--shards", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.shards <= 120:
        parser.error("--shards must be between 1 and 120")
    args.root.mkdir(parents=True, exist_ok=True)
    model_dir = args.root / "model"
    patterns = ["*.json", "*.jinja", "*.txt", "*.safetensors", "LICENSE", "README.md"]
    snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=model_dir, allow_patterns=patterns, max_workers=4)
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
    model_config.update(model_type="maple", mask_token_id=tokenizer.mask_token_id)
    model_config.pop("auto_map", None)
    config_dir = args.root / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(model_config, indent=2) + "\n")
    manifest = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": DATA_ID,
        "dataset_revision": DATA_REVISION,
        "paper_corpus": False,
        "mask_token_id": tokenizer.mask_token_id,
        "max_length": args.max_length,
        "requested_shards": args.shards,
    }
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
    print("Pinned snapshots downloaded", flush=True)
    stats = {"train_samples": 0, "train_tokens": 0, "validation_samples": 0, "skipped": 0}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init_worker,
        initargs=(str(args.root / "tokenizer"),),
    ) as pool:
        for file, counts in pool.map(prepare_shard, [(args.root, file, args.max_length) for file in files]):
            for key, value in counts.items():
                stats[key] += value
            print(json.dumps({"prepared": file, **stats}), flush=True)
    manifest.update(stats)
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
