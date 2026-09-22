"""Decode an exported Maple I-DLM checkpoint with exact strided sampling."""

import argparse
import json

import torch
from transformers import AutoTokenizer

from veomni.models.transformers.maple.runtime import introspective_generate, load_maple_for_inference
from veomni.utils.device import get_device_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--prompt", default="Explain why the sky is blue.")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true", help="Recompute the prefix for reference comparisons")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, fix_mistral_regex=False)
    with load_maple_for_inference(args.checkpoint, attention="sdpa") as model:
        inputs = tokenizer.apply_chat_template(
            [dict(role="user", content=args.prompt)], add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"].to(get_device_type())
        result = introspective_generate(
            model,
            inputs,
            mask_token_id=tokenizer.mask_token_id,
            max_new_tokens=args.tokens,
            stride=args.stride,
            temperature=args.temperature,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=not args.no_cache,
        )
    print(tokenizer.decode(result.sequences[0, inputs.shape[1] :], skip_special_tokens=False))
    print(
        json.dumps(
            dict(
                proposed=result.proposed,
                accepted=result.accepted,
                forward_passes=result.forward_passes,
                processed_tokens=result.processed_tokens,
                use_cache=not args.no_cache,
            )
        )
    )


if __name__ == "__main__":
    main()
