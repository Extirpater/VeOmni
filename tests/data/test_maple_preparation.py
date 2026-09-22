"""Preparation directories must describe exactly one pinned tokenization."""

import json
import runpy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from veomni.data import build_dataloader, build_dataset
from veomni.data.maple import conversation_split, encode_conversation, process_maple_conversation, raw_shard_identity


preparation = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/maple_idlm/prepare_data.py"))
validate_preparation_root = preparation["validate_preparation_root"]


@pytest.fixture
def prepared_root(tmp_path):
    manifest = dict(model_revision="model", dataset_revision="data", max_length=4096, requested_shards=2)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for split in ("train", "validation"):
        (tmp_path / split).mkdir()
        for index in range(2):
            (tmp_path / split / f"train-{index:05d}-of-00120.parquet").touch()
    return tmp_path, manifest


@pytest.mark.parametrize("key,value", [("requested_shards", 1), ("max_length", 1024), ("dataset_revision", "other")])
def test_refuse_mixed_preparation(prepared_root, key, value):
    root, manifest = prepared_root
    original = (root / "manifest.json").read_text()
    with pytest.raises(ValueError, match="choose a new --root"):
        validate_preparation_root(root, {**manifest, key: value})
    assert (root / "manifest.json").read_text() == original
    assert len(list((root / "train").glob("*.parquet"))) == 2


@pytest.mark.parametrize("name", ["train-00002-of-00120.parquet", "backup/train-00000-of-00120.parquet", "old.jsonl"])
def test_refuse_unmanifested_shards(prepared_root, name):
    root, manifest = prepared_root
    extra = root / "train" / name
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.touch()
    with pytest.raises(ValueError, match="Unexpected shards"):
        validate_preparation_root(root, manifest)


def test_resume_identical_preparation_or_empty_root(prepared_root, tmp_path):
    root, manifest = prepared_root
    validate_preparation_root(root, manifest)
    # An interrupted job may have completed only some of its requested shards.
    (root / "train/train-00001-of-00120.parquet").unlink()
    validate_preparation_root(root, manifest)
    validate_preparation_root(tmp_path / "new", manifest)


def test_existing_tokens_require_provenance(prepared_root):
    root, manifest = prepared_root
    (root / "manifest.json").unlink()
    with pytest.raises(ValueError, match="no manifest"):
        validate_preparation_root(root, manifest)


def test_local_model_identity_tracks_assets_and_requires_all_shards(tmp_path):
    index = {"weight_map": {"model.weight": "model-00001.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    weight = tmp_path / "model-00001.safetensors"
    weight.write_bytes(b"local weights")
    revision = preparation["local_model_revision"](tmp_path)
    assert revision.startswith("local-")
    assert preparation["local_model_revision"](tmp_path) == revision
    (tmp_path / "config.json").write_text('{"quantize": true}')
    assert preparation["local_model_revision"](tmp_path) != revision
    weight.unlink()
    with pytest.raises(ValueError, match="missing a weight shard"):
        preparation["local_model_revision"](tmp_path)


class CharacterTokenizer:
    def __init__(self, require_worker=False):
        self.require_worker = require_worker

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}!" for m in messages)

    def encode(self, text, **kwargs):
        if self.require_worker:
            assert torch.utils.data.get_worker_info() is not None
        return list(text.encode())


def conversation(prompt="question", answer="answer"):
    return [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]


def test_online_masking_truncation_and_holdout():
    tokenizer = CharacterTokenizer()
    source = conversation(answer="reasoning followed by answer")
    encoded = encode_conversation(source, tokenizer, 32)
    expected = b"<user>question!<assistant>reason"
    assert encoded["input_ids"] == list(expected)
    assert encoded["labels"] == [-100] * 15 + list(expected[15:])
    split = conversation_split(source)
    [actual] = process_maple_conversation({"conversations": source}, tokenizer, 32, split=split)
    assert actual["input_ids"].tolist() == encoded["input_ids"]
    assert actual["labels"].tolist() == encoded["labels"]
    assert actual["attention_mask"].tolist() == [1] * 32
    opposite = "validation" if split == "train" else "train"
    assert process_maple_conversation({"conversations": source}, tokenizer, 32, split=opposite) == []
    assert conversation_split(conversation(answer="a different response")) == split
    normalized = [{"role": "user", "content": "question"}, {"role": "assistant", "content": source[1]["value"]}]
    assert encode_conversation(normalized, tokenizer, 32) == encoded
    assert encode_conversation(conversation(prompt="p" * 40), tokenizer, 32) is None


def test_online_mode_cannot_reuse_offline_root(prepared_root):
    root, manifest = prepared_root
    with pytest.raises(ValueError, match="tokenization mode"):
        validate_preparation_root(root, {**manifest, "tokenization": "online"})


def test_raw_shard_identity_rejects_partial_or_extra_data(tmp_path):
    root = tmp_path / "raw/data"
    root.mkdir(parents=True)
    shard = root / "train-00000-of-00120.parquet"
    shard.write_bytes(b"raw")
    original = raw_shard_identity(tmp_path, 1)
    shard.write_bytes(b"changed raw")
    assert raw_shard_identity(tmp_path, 1) != original
    with pytest.raises(ValueError, match="Raw shards"):
        raw_shard_identity(tmp_path, 2)
    (root / "untracked.jsonl").touch()
    with pytest.raises(ValueError, match="Raw shards"):
        raw_shard_identity(tmp_path, 1)


def test_online_native_workers_and_checkpoint_resume(tmp_path, monkeypatch):
    import veomni.data.data_collator as collators
    import veomni.data.data_loader as loaders

    state = SimpleNamespace(dp_size=1, dp_rank=0, sp_enabled=False, sp_size=1, sp_rank=0)
    monkeypatch.setattr(loaders, "get_parallel_state", lambda: state)
    monkeypatch.setattr(collators, "get_parallel_state", lambda: state)
    rows = [{"conversations": conversation(str(index), "response " * 12)} for index in range(80)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "raw.parquet")
    dataset = build_dataset(
        "mapping",
        train_path=str(tmp_path / "raw.parquet"),
        transform=partial(
            process_maple_conversation, tokenizer=CharacterTokenizer(require_worker=True), max_seq_len=64
        ),
        seed=7,
    )

    def make_loader():
        return build_dataloader(
            "native",
            dataset=dataset,
            micro_batch_size=1,
            global_batch_size=2,
            dataloader_batch_size=1,
            max_seq_len=64,
            train_steps=12,
            dyn_bsz=True,
            dyn_bsz_buffer_size=2,
            bsz_warmup_ratio=0,
            num_workers=2,
            prefetch_factor=2,
            pin_memory=False,
            seed=7,
        )

    loader = make_loader()
    iterator = iter(loader)
    for _ in range(3):
        next(iterator)
    snapshot = loader.state_dict()
    expected = [next(iterator) for _ in range(4)]
    restored = make_loader()
    restored.load_state_dict(snapshot)
    resumed = iter(restored)
    for expected_step in expected:
        actual_step = next(resumed)
        assert len(actual_step) == len(expected_step)
        for actual, reference in zip(actual_step, expected_step):
            for key in ("input_ids", "labels", "attention_mask", "position_ids"):
                torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)


def test_online_evaluation_samples_only_heldout_rows(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    evaluation = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/maple_idlm/evaluate.py"))
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    source = [conversation(str(index), "heldout answer") for index in range(1000)]
    heldout = [row for row in source if conversation_split(row) == "validation"]
    assert len(heldout) > 2
    directory = tmp_path / "raw/data"
    directory.mkdir(parents=True)
    shard = directory / "train-00000-of-00120.parquet"
    pq.write_table(pa.Table.from_pylist([{"conversations": row} for row in source]), shard)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(tokenization="online", max_length=64, requested_shards=1, raw_shards=raw_shard_identity(tmp_path, 1))
        )
    )
    actual = evaluation["validation_sample"](tmp_path, 2, 7)
    assert actual == evaluation["validation_sample"](tmp_path, 2, 7)
    assert len(actual) == 2
    expected = [encode_conversation(row, tokenizer, 64) for row in heldout]
    assert all(row in expected for row in actual)
    extra = directory / "extra.parquet"
    extra.touch()
    with pytest.raises(ValueError, match="Raw shards"):
        evaluation["validation_sample"](tmp_path, 2, 7)
    extra.unlink()
    shard.write_bytes(shard.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Raw dataset changed"):
        evaluation["validation_sample"](tmp_path, 2, 7)
