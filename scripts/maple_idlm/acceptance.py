"""Measure ISD acceptance and tokens per forward on held-out prompts."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from veomni.models.transformers.maple.runtime import introspective_generate, load_maple_for_inference
from veomni.utils.device import get_device_type, synchronize


def main():
    from evaluate import validation_sample

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, fix_mistral_regex=False)
    mask_token_id = json.loads((args.data_root / "manifest.json").read_text())["mask_token_id"]
    prompts = []
    for record in validation_sample(args.data_root, args.prompts, args.seed):
        labels = torch.as_tensor(record["labels"])
        supervised = torch.nonzero(labels != -100)
        # Keep records with a nonempty prompt before the first supervised token.
        if supervised.numel() and int(supervised[0]) > 0:
            prompts.append(torch.as_tensor(record["input_ids"])[: int(supervised[0])])
    results = {}
    with load_maple_for_inference(args.checkpoint, attention="sdpa") as model:
        for stride in args.strides:
            totals = dict(proposed=0, accepted=0, forward_passes=0, new_tokens=0, seconds=0.0)
            for index, prompt in enumerate(prompts):
                generator = torch.Generator(device=get_device_type()).manual_seed(args.seed + index)
                ids = prompt[None].to(get_device_type())
                synchronize()
                start = time.perf_counter()
                result = introspective_generate(
                    model,
                    ids,
                    mask_token_id=mask_token_id,
                    max_new_tokens=args.tokens,
                    stride=stride,
                    temperature=args.temperature,
                    eos_token_id=tokenizer.eos_token_id,
                    generator=generator,
                    use_cache=True,
                )
                synchronize()
                totals["seconds"] += time.perf_counter() - start
                totals["proposed"] += result.proposed
                totals["accepted"] += result.accepted
                totals["forward_passes"] += result.forward_passes
                totals["new_tokens"] += result.sequences.shape[1] - ids.shape[1]
            totals["acceptance"] = totals["accepted"] / max(1, totals["proposed"])
            totals["tokens_per_forward"] = totals["new_tokens"] / max(1, totals["forward_passes"])
            totals["tokens_per_second"] = totals["new_tokens"] / max(1e-9, totals["seconds"])
            results[f"stride_{stride}"] = totals
            print(json.dumps({f"stride_{stride}": totals}), flush=True)
    output = dict(
        checkpoint=args.checkpoint, prompts=len(prompts), tokens=args.tokens, temperature=args.temperature, **results
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
