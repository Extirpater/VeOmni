"""Evaluate a reproducible held-out sample with the training quantizer enabled."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch

from veomni.models.transformers.maple.runtime import (
    initialize_mask_token,
    load_maple_for_inference,
    prepare_idlm_inputs,
)
from veomni.utils.device import get_device_type


def validation_sample(root, count, seed):
    from veomni.data.maple import conversation_split, encode_conversation, raw_shard_identity

    manifest = json.loads((root / "manifest.json").read_text())
    online = manifest.get("tokenization", "offline") == "online"
    if online and raw_shard_identity(root, manifest["requested_shards"]) != manifest["raw_shards"]:
        raise ValueError("Raw dataset changed since preparation; prepare a new root")
    rng = random.Random(seed)
    selected, seen = [], 0
    directory = root / "raw" / "data" if online else root / "validation"
    for path in sorted(directory.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if online and conversation_split(row.get("conversations", row.get("messages"))) != "validation":
                    continue
                seen += 1
                index = len(selected) if len(selected) < count else rng.randrange(seen)
                if len(selected) < count:
                    selected.append(row)
                elif index < count:
                    selected[index] = row
    if online:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", fix_mistral_regex=False)
        selected = [
            encode_conversation(row.get("conversations", row.get("messages")), tokenizer, manifest["max_length"])
            for row in selected
        ]
        selected = [row for row in selected if row is not None]
    return selected


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--data-root", type=Path, default=Path("outputs/maple_idlm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--initialize-mask", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.data_root / "manifest.json").read_text())
    if not (manifest.get("data_ready") or manifest.get("train_samples")) or args.samples < 1:
        raise ValueError("Complete preparation first and request a positive sample count")
    records = validation_sample(args.data_root, args.samples, args.seed)
    if not records:
        raise ValueError("No held-out samples")
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    totals = dict(masked_loss_sum=0.0, masked_tokens=0, clean_loss_sum=0.0, clean_tokens=0)
    with load_maple_for_inference(
        args.checkpoint,
        config=args.config,
        config_kwargs=dict(
            idlm_enabled=True, idlm_block_size=args.block_size, mask_token_id=manifest["mask_token_id"]
        ),
    ) as model:
        if args.initialize_mask:
            initialize_mask_token(model, manifest["mask_token_id"], seed=args.seed)
        for index, record in enumerate(records):
            length = min(len(record["input_ids"]), manifest["max_length"])
            ids = torch.zeros((1, manifest["max_length"]), dtype=torch.long, device=get_device_type())
            labels = torch.full_like(ids, -100)
            valid = torch.zeros_like(ids)
            ids[0, :length] = torch.tensor(record["input_ids"][:length], device=ids.device)
            labels[0, :length] = torch.tensor(record["labels"][:length], device=ids.device)
            valid[0, :length] = 1
            positions = torch.arange(ids.shape[-1], device=ids.device)[None]
            output = model(input_ids=ids, labels=labels, attention_mask=valid, position_ids=positions)
            _, _, noisy, clean = prepare_idlm_inputs(ids, labels, positions, manifest["mask_token_id"])
            for branch, targets in (("masked", noisy), ("clean", clean)):
                count = int((targets != -100).sum())
                loss = output.aux_metrics[f"idlm_{branch}_ce"]
                if not loss.isfinite():
                    raise FloatingPointError(f"Nonfinite {branch} validation loss")
                totals[f"{branch}_loss_sum"] += float(loss) * count
                totals[f"{branch}_tokens"] += count
            print(json.dumps(dict(evaluated=index + 1, **totals)), flush=True)
    result = dict(
        checkpoint=args.checkpoint,
        samples=len(records),
        sample_sha256=fingerprint,
        seed=args.seed,
        block_size=args.block_size,
        dataset_revision=manifest["dataset_revision"],
        **totals,
    )
    for branch in ("masked", "clean"):
        result[f"{branch}_ce"] = totals[f"{branch}_loss_sum"] / totals[f"{branch}_tokens"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
