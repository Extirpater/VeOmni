"""Maple's OpenThoughts response masking for offline and worker-side tokenization."""

import hashlib

import torch


def raw_shard_identity(root, requested_shards):
    """Validate the raw directory used by the loader and fingerprint its files."""
    directory = root / "raw" / "data"
    expected = {f"train-{i:05d}-of-00120.parquet" for i in range(requested_shards)}
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Raw shards do not match the manifest; finish the download in a separate root")
    return {
        name: {"bytes": (directory / name).stat().st_size, "mtime_ns": (directory / name).stat().st_mtime_ns}
        for name in sorted(expected)
    }


def conversation_split(conversation):
    """Keep repeated prompts on the same side of the fixed 1% holdout."""
    prompt = conversation[0].get("value", conversation[0].get("content", ""))
    digest = hashlib.sha256(prompt.encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 100 == 0 else "train"


def encode_conversation(conversation, tokenizer, max_length):
    messages = [
        {
            "role": {"human": "user", "gpt": "assistant"}.get(m.get("from"), m.get("role")),
            "content": m.get("value", m.get("content", "")),
        }
        for m in conversation
    ]
    ids, labels = [], []
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
    ids, labels = ids[:max_length], labels[:max_length]
    if len(ids) < 2 or sum(label != -100 for label in labels[1:]) < 2:
        return None
    return {"input_ids": ids, "labels": labels}


def process_maple_conversation(example, tokenizer, max_seq_len, split="train", **kwargs):
    """Tokenize raw rows in VeOmni data workers, excluding held-out/empty responses."""
    conversation = example.get("conversations", example.get("messages"))
    if not conversation:
        raise ValueError("OpenThoughts row has no conversation field")
    if conversation_split(conversation) != split:
        return []
    encoded = encode_conversation(conversation, tokenizer, max_seq_len)
    if encoded is None:
        return []
    result = {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    return [result]
