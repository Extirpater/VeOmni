"""Maple launch validation and final checkpoints must preserve resume state."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from tasks.train_maple_idlm import save_final_checkpoint
from veomni.checkpoint import layout


launch = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/maple_idlm/launch.py"))


def make_checkpoint(root, step, world_size=1):
    checkpoint = root / "checkpoints" / f"global_step_{step}"
    weights = checkpoint / "model/ckpt"
    weights.mkdir(parents=True)
    (weights / ".metadata").touch()
    for directory in ("loader", "extra_state"):
        (checkpoint / directory).mkdir()
        for rank in range(world_size):
            (checkpoint / directory / f"rank_{rank}.pt").touch()
    layout.write_manifest(str(checkpoint), global_step=step, world_size=world_size)
    return checkpoint


def test_auto_resume_requires_completed_checkpoint(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("old metrics\n")
    with pytest.raises(ValueError, match="no complete checkpoint"):
        launch["resolve_resume_checkpoint"](tmp_path, "auto", 1)
    completed = make_checkpoint(tmp_path, 10)
    (tmp_path / "checkpoints/global_step_20").mkdir()
    assert launch["resolve_resume_checkpoint"](tmp_path, "auto", 1) == str(completed)
    assert (tmp_path / "metrics.jsonl").read_text() == "old metrics\n"


def test_resume_requires_matching_world_size_and_rank_cursors(tmp_path):
    checkpoint = make_checkpoint(tmp_path, 10, world_size=2)
    with pytest.raises(ValueError, match="same GPU count"):
        launch["resolve_resume_checkpoint"](tmp_path, "auto", 1)
    (checkpoint / "loader/rank_1.pt").unlink()
    with pytest.raises(ValueError, match="missing loader/rank_1"):
        launch["resolve_resume_checkpoint"](tmp_path, str(checkpoint), 2)


def test_optimizer_flags_require_dcp():
    from veomni.arguments import CheckpointConfig

    config = CheckpointConfig(save_optimizer=False, load_optimizer=False)
    assert not config.save_optimizer and not config.load_optimizer
    with pytest.raises(ValueError, match="require the dcp"):
        CheckpointConfig(manager="bcp", save_optimizer=False)


@pytest.mark.parametrize("mixed_precision, required_gib", [(False, 80), (True, 160), (None, 160)])
def test_prepare_only_never_starts_training_or_accepts_stale_data(
    tmp_path, monkeypatch, mixed_precision, required_gib
):
    manifest = dict(train_samples=2, train_tokens=8, requested_shards=1, max_length=4)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for split in ("train", "validation"):
        (tmp_path / split).mkdir()
        (tmp_path / split / "train-00000-of-00120.parquet").touch()
    config = {
        "model": {},
        "data": {"max_seq_len": 4},
        "train": {
            "global_batch_size": 8,
            "micro_batch_size": 1,
            "checkpoint": {"load_path": None, "save_optimizer": False},
        },
    }
    if mixed_precision is not None:
        config["model"]["accelerator"] = {"fsdp_config": {"mixed_precision": {"enable": mixed_precision}}}
    config_file = tmp_path / "input.yaml"
    config_file.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(
        root=tmp_path,
        config=config_file,
        gpus=8,
        run_name="run",
        prepare_only=True,
        steps=None,
        init_weights=None,
    )
    target = launch["prepare_run"](args)
    resolved = yaml.safe_load(target.read_text())
    assert resolved["data"]["train_size"] == 8
    assert resolved["train"]["checkpoint"]["load_path"] is None
    assert target.parent == tmp_path / "run"
    assert resolved["model"]["model_path"] == str(tmp_path / "model")
    assert not (target.parent / "training.log").exists()
    args.prepare_only = False
    monkeypatch.setenv("MAPLE_KEEP_CHECKPOINTS", "1")
    monkeypatch.setattr(launch["shutil"], "disk_usage", lambda _: SimpleNamespace(free=(required_gib - 1) * 1024**3))
    with pytest.raises(ValueError, match=f"Need {required_gib} GiB"):
        launch["prepare_run"](args)
    monkeypatch.setattr(launch["shutil"], "disk_usage", lambda _: SimpleNamespace(free=(required_gib + 1) * 1024**3))
    assert launch["prepare_run"](args) == target
    (tmp_path / "train/train-00001-of-00120.parquet").touch()
    with pytest.raises(ValueError, match="shards do not match"):
        launch["prepare_run"](args)


def test_launch_rejects_changed_local_checkpoint(tmp_path):
    manifest = dict(train_samples=1, model_revision="local-stale", model_path=str(tmp_path / "model"))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "model").mkdir()
    (tmp_path / "model/model.safetensors.index.json").write_text('{"weight_map": {"w": "weights.safetensors"}}')
    (tmp_path / "model/weights.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Local model assets changed"):
        launch["prepare_run"](SimpleNamespace(root=tmp_path))
    (tmp_path / "config").mkdir()
    (tmp_path / "config/config.json").write_text("{}")
    result = subprocess.run(
        [sys.executable, str(Path(launch["__file__"])), "--root", str(tmp_path), "--prepare-only"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "Local model assets changed" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_online_launch_requires_complete_unchanged_raw_data(tmp_path):
    from veomni.data.maple import raw_shard_identity

    directory = tmp_path / "raw/data"
    directory.mkdir(parents=True)
    shard = directory / "train-00000-of-00120.parquet"
    shard.write_bytes(b"raw parquet")
    manifest = dict(
        tokenization="online",
        data_ready=True,
        raw_samples=100,
        raw_shards=raw_shard_identity(tmp_path, 1),
        requested_shards=1,
        max_length=4096,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config = {
        "model": {},
        "data": {"max_seq_len": 4096, "train_size": 9000},
        "train": {"global_batch_size": 32, "micro_batch_size": 1, "checkpoint": {"load_path": None}},
    }
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(
        root=tmp_path, config=path, gpus=8, run_name="run", prepare_only=True, steps=None, init_weights=None
    )
    target = launch["prepare_run"](args)
    resolved = yaml.safe_load(target.read_text())
    assert resolved["data"]["train_path"] == str(directory)
    assert resolved["data"]["data_type"] == "conversation"
    assert resolved["data"]["train_size"] == 9000
    assert not (tmp_path / "train").exists()
    assert not (target.parent / "training.log").exists()
    shard.write_bytes(b"changed raw parquet")
    with pytest.raises(ValueError, match="Raw dataset changed"):
        launch["prepare_run"](args)
    manifest["data_ready"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="has not finished"):
        launch["prepare_run"](args)


@pytest.mark.parametrize("already_saved", [False, True])
def test_deadline_saves_both_halves_before_waiting(already_saved):
    events = []
    state = SimpleNamespace(global_step=17)
    previous = 17 if already_saved else 10
    checkpoint = SimpleNamespace(_last_dcp_step=previous, _save_dcp=Mock(side_effect=lambda _: events.append("model")))
    global_state = SimpleNamespace(
        _last_saved_step=previous, save_global_state=Mock(side_effect=lambda _: events.append("cursor"))
    )
    trainer = SimpleNamespace(
        state=state,
        checkpoint_callback=checkpoint,
        global_state_callback=global_state,
        wait_for_pending_save=lambda: events.append("wait"),
    )
    save_final_checkpoint(trainer)
    assert events == (["wait"] if already_saved else ["model", "cursor", "wait"])


def test_init_weights_starts_a_stage_from_an_export(tmp_path):
    from veomni.data.maple import raw_shard_identity

    directory = tmp_path / "raw/data"
    directory.mkdir(parents=True)
    (directory / "train-00000-of-00120.parquet").write_bytes(b"raw parquet")
    manifest = dict(
        tokenization="online",
        data_ready=True,
        raw_samples=100,
        raw_shards=raw_shard_identity(tmp_path, 1),
        requested_shards=1,
        max_length=4096,
    )
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    export = tmp_path / "export"
    export.mkdir()
    config = {
        "model": {"model_config": {"idlm_initialize_mask_token": True}},
        "data": {"max_seq_len": 4096, "train_size": 9000},
        "train": {"global_batch_size": 32, "micro_batch_size": 1, "checkpoint": {"load_path": None}},
    }
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(config))
    args = SimpleNamespace(
        root=tmp_path, config=path, gpus=8, run_name="stage2", prepare_only=True, steps=None, init_weights=export
    )
    with pytest.raises(ValueError, match="exported HF checkpoint"):
        launch["prepare_run"](args)
    (export / "config.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="idlm_initialize_mask_token: false"):
        launch["prepare_run"](args)
    config["model"]["model_config"]["idlm_initialize_mask_token"] = False
    path.write_text(yaml.safe_dump(config))
    resolved = yaml.safe_load(launch["prepare_run"](args).read_text())
    assert resolved["model"]["model_path"] == str(export.resolve())
