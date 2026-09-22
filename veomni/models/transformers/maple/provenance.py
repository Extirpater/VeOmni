"""Local Maple checkpoint identity used by preparation and launch."""

import hashlib
import json


def local_model_revision(model_dir):
    """Fingerprint local assets and weight file identities without copying them."""
    index = model_dir / "model.safetensors.index.json"
    weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    if not weights:
        raise ValueError("Local checkpoint index has no weight shards")
    identity = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        path = model_dir / name
        identity[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    for name in weights:
        path = (model_dir / name).resolve()
        if path.parent != model_dir or not path.is_file():
            raise ValueError(f"Local checkpoint is missing a weight shard or has an invalid path: {name}")
        info = path.stat()
        identity[name] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return f"local-{digest}"
