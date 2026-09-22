"""VeOmni Maple run with finite-gradient checks and a resumable wall-time stop."""

import json
import math
import os
import shutil
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from veomni.arguments import VeOmniArguments, parse_args
from veomni.checkpoint.layout import checkpoint_is_complete
from veomni.data.maple import process_maple_conversation
from veomni.trainer.callbacks.base import Callback
from veomni.trainer.text_trainer import TextTrainer
from veomni.utils.device import get_device_type


class WallTimeReached(Exception):
    pass


class MapleTrainer(TextTrainer):
    def _build_data_transform(self):
        args = self.base.args
        if args.data.data_type == "conversation":
            self.base.data_transform = partial(
                process_maple_conversation,
                tokenizer=self.base.model.tokenizer,
                max_seq_len=args.data.max_seq_len,
            )
        else:
            super()._build_data_transform()


def save_final_checkpoint(trainer):
    """Save both halves on every rank even when the deadline misses the cadence."""
    state = trainer.state
    checkpoint = trainer.checkpoint_callback
    global_state = trainer.global_state_callback
    if checkpoint._last_dcp_step != state.global_step:
        checkpoint._save_dcp(state)
    if global_state._last_saved_step != state.global_step:
        global_state.save_global_state(state)
    trainer.wait_for_pending_save()


class RunMetrics(Callback):
    def __init__(self, trainer):
        super().__init__(trainer)
        self.start = None
        self.deadline = None
        self.limit = float(os.environ.get("MAPLE_RUN_SECONDS", "43200"))
        self.keep_checkpoints = int(os.environ.get("MAPLE_KEEP_CHECKPOINTS", "1"))
        if self.keep_checkpoints < 1:
            raise ValueError("MAPLE_KEEP_CHECKPOINTS must retain at least one complete checkpoint")
        self.path = Path(trainer.args.train.checkpoint.output_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    def on_train_begin(self, state, **kwargs):
        model = self.trainer.model.model
        if model.config.idlm_initialize_mask_token and not self.trainer.args.train.checkpoint.load_path:
            from veomni.models.transformers.maple.runtime import initialize_mask_token

            norms = initialize_mask_token(model, model.config.mask_token_id, seed=self.trainer.args.train.seed)
            if dist.get_rank() == 0:
                (self.path / "mask_initialization.json").write_text(
                    json.dumps(dict(token_id=model.config.mask_token_id, row_norms=norms))
                )

    def on_step_begin(self, state, **kwargs):
        if self.start is None:
            clock = self.path / "training_clock.json"
            if clock.exists():
                saved = json.loads(clock.read_text())
                self.start, self.deadline = saved["start"], saved["deadline"]
            else:
                self.start = time.time()
                self.deadline = self.start + self.limit
                if dist.get_rank() == 0:
                    temporary = clock.with_suffix(".tmp")
                    temporary.write_text(json.dumps(dict(start=self.start, deadline=self.deadline)))
                    temporary.replace(clock)

    def prune_checkpoints(self):
        if dist.get_rank() != 0:
            return
        checkpoints = Path(self.trainer.args.train.checkpoint.save_path)
        complete = sorted(
            (
                path
                for path in checkpoints.glob("global_step_*")
                if path.name[12:].isdigit() and checkpoint_is_complete(str(path))
            ),
            key=lambda path: int(path.name[12:]),
        )
        # Only prune this run's older, completed checkpoints.
        for path in complete[: -self.keep_checkpoints]:
            shutil.rmtree(path)

    def on_step_end(self, state, **kwargs):
        record = {"time": time.time(), "step": state.global_step, "epoch": state.epoch}
        record.update({key: float(value) for key, value in self.trainer.step_env_metrics.items()})
        record["elapsed_seconds"] = time.time() - self.start
        if dist.get_rank() == 0:
            with (self.path / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.prune_checkpoints()
        flag = torch.tensor(int(time.time() >= self.deadline), device=get_device_type())
        dist.broadcast(flag, src=0)
        if flag.item():
            raise WallTimeReached

    def on_train_end(self, state, **kwargs):
        self.prune_checkpoints()
        if dist.get_rank() == 0:
            (self.path / "completed.json").write_text(json.dumps(dict(time=time.time(), step=state.global_step)))


def main():
    args = parse_args(VeOmniArguments)
    if args.model.ops_implementation.qat_implementation != "ternary":
        raise ValueError("This entry point requires ternary QAT")
    trainer = MapleTrainer(args)
    original_postforward = trainer.base.postforward

    def finite_postforward(outputs, micro_batch):
        if not torch.isfinite(outputs.loss).all():
            raise FloatingPointError("Nonfinite objective; backward and optimizer step were not applied")
        return original_postforward(outputs, micro_batch)

    trainer.base.postforward = finite_postforward
    original_clip = trainer.base.model.clip_grad_norm

    def finite_clip():
        norm = original_clip()
        if not math.isfinite(float(norm)):
            raise FloatingPointError("Nonfinite global gradient norm; optimizer step was not applied")
        return norm

    trainer.base.model.clip_grad_norm = finite_clip
    trainer.base._callbacks.append(RunMetrics(trainer.base))
    try:
        trainer.train()
    except WallTimeReached:
        # All ranks stop between optimizer steps and join the ordinary checkpoint
        # and cursor callbacks. Neither a partial optimizer step nor SIGKILL.
        save_final_checkpoint(trainer.base)
        trainer.on_epoch_end()
        trainer.on_train_end()
        trainer.base.destroy_distributed()


if __name__ == "__main__":
    main()
