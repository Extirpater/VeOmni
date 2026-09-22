# Maple introspective diffusion training

Convert [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview)
using the dual noisy/clean objective and strided decoder from
[Introspective Diffusion Language Models](https://arxiv.org/abs/2604.11035), with
full-parameter ternary QAT in VeOmni's text trainer.

## Run

On a prepared GPU host:

```bash
./start_maple.sh
```

Starts a fresh 12-hour run from the original Maple weights. Logs, metrics and
checkpoints go to `outputs/maple_idlm/run-fresh-<timestamp>/`. W&B uses your
saved login and the `maple-idlm` project.

For a new host, prepare the environment and data first:

```bash
uv sync --frozen --extra gpu --dev --python 3.12
.venv/bin/wandb login
.venv/bin/python scripts/maple_idlm/prepare_data.py --workers 16
./start_maple.sh
```

The default config targets **one H100 80 GB with CPU offload**. Allow at least
220 GiB available host RAM and roughly 600 GB free disk before setup. Each full
checkpoint is about 151 GiB. The launcher requires 320 GiB free for a fresh run.
Training keeps one completed checkpoint per run; `MAPLE_KEEP_CHECKPOINTS`
overrides this. Older runs are untouched.

For a GPU node, copy `configs/text/maple_idlm.yaml`, disable
`model.accelerator.fsdp_config.offload`, and set `train.global_batch_size` to a
multiple of GPUs × microbatch size. Keep recompute enabled initially. Then use:

```bash
.venv/bin/python scripts/maple_idlm/launch.py --config YOUR_CONFIG.yaml --gpus 8
```

The launcher runs `torchrun` in the foreground with output in `training.log`.
`--prepare-only` writes the config without training. Multi-GPU performance still
needs validation. Set `OMP_NUM_THREADS` to override the default CPU thread limit.

To resume, set `train.checkpoint.load_path: auto` and supply the existing
`--run-name`. Its deadline survives restart; `--fresh-window` starts a new time
window. Fresh training uses `load_path: null` and rejects reused training state.
A lock prevents concurrent launches against the same data root. W&B and
`metrics.jsonl` record training progress. At the deadline, training stops between
updates and saves a checkpoint. Nonfinite loss or gradients abort before an update.

## Data, objective and kernels

Preparation downloads pinned Maple weights and all 120 OpenThoughts3 shards,
recording revisions in `manifest.json`. It applies Maple's chat template,
supervises responses including reasoning, and keeps the first 4096 tokens.
A prompt-hashed 1% validation split stays out of training. `--shards N --root PATH`
prepares a smaller experiment; preparation can restart. OpenThoughts is the
authors' public example dataset, not the paper's original corpus.

Training concatenates `[noisy | clean]` with duplicated logical positions. Noisy
queries see causal tokens in their own block and clean tokens in earlier blocks;
clean queries see only their causal clean prefix. Packed documents and padding
are isolated, and labels shift once. Loss is masked cross entropy plus
`idlm_clean_weight * clean_cross_entropy`. The example uses block size 1 and
clean weight 0.2, with ten warmup updates. This pilot does not reproduce the
paper's full curriculum: later stages use block sizes 2 and 3 and auto-balanced
loss. The matching decoding stride is block size + 1.

FlexAttention implements the training mask and backward. Triton grouped GEMM
runs the experts with Maple's clamped SwiGLU; Liger supplies linear cross entropy
and RMSNorm. FSDP2 data sharding is supported; SP/EP sizes greater than one are
rejected. Maple's QK normalization, partial RoPE and global NoPE are preserved.

Ternary QAT recomputes absmax scales per 128 input features and quantizes
attention/expert weights to `{-s, 0, s}` with an identity STE. Routers, embeddings,
norms and the language head stay floating point. CUDA uses a Triton quantizer
and BF16 GEMMs. BF16 masters, gradients and AnyPrecisionAdamW states still need
floating-point training memory. The public ternary checkpoint is the starting
point; its unpublished latent masters are unavailable.

The new `<|idlm_mask|>` token uses an unused padded vocabulary row. Fresh training
initializes its input/output rows with the vocabulary mean plus seeded noise;
resume preserves them. Preparation writes the registered Maple config without
executing downloaded model code.

## Export and evaluate

Replace `RUN` and `STEP` with an actual run directory and completed step:

```bash
.venv/bin/python scripts/merge_dcp_to_hf.py \
  --load-dir outputs/maple_idlm/RUN/checkpoints/global_step_STEP \
  --save-dir outputs/maple_idlm/export \
  --model-assets-dir outputs/maple_idlm/RUN/model_assets
.venv/bin/torchrun --standalone --nproc-per-node=1 scripts/maple_idlm/generate.py \
  outputs/maple_idlm/export --stride 2 --tokens 128
.venv/bin/torchrun --standalone --nproc-per-node=1 scripts/maple_idlm/evaluate.py \
  outputs/maple_idlm/export --output outputs/maple_idlm/validation.json
```

Exports contain floating QAT masters plus configuration and tokenizer assets.
Use the provided decoder so the same ternary quantization applies on every
forward. When starting a new training run from an export, set
`idlm_initialize_mask_token: false` to preserve its trained mask rows.

The decoder uses exact `p/q` acceptance with residual correction, causal/sliding
SDPA and KV rollback; `--no-cache` provides a reference. It preserves the
converted model's AR distribution, not necessarily the original Maple model's
quality. It does not implement the paper's serving system or claim its throughput.

Validation uses a fixed 64-example held-out sample and reports both losses.
Maple FLOPs accounting is not registered: logged zero MFU is not a measurement.

Run the focused tests with:

```bash
.venv/bin/pytest -q tests/models/test_maple_idlm.py
```

Run these tests manually; the existing GPU/NPU CI workflows do not select them.

Implementation notes:

- `maple/runtime.py` contains the training masks, decoder, cache and inference
  setup. The checkpoint exporter uses VeOmni's configuration registry to preserve
  Maple's model assets.
- Single-GPU CPU offload still needs FSDP2 hooks. AnyPrecision resume allocates
  optimizer state directly; a synthetic gradient step can exhaust host RAM.
- Cache rollback crops KV tensors and their position/padding metadata together.
  Sliding layers retain their history for rollback. With the pinned Transformers
  version, negative `crop` values remove trailing tokens; reset clears the length.
- Fully masked SDPA rows get a finite dummy key, then their outputs are zeroed.
  This prevents invalid padding outputs and gradients on the cuDNN backend.
