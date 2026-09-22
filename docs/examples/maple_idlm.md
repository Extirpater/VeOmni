# Maple introspective diffusion training

Convert a local Maple checkpoint, or [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview),
using the dual noisy/clean objective and strided decoder from
[Introspective Diffusion Language Models](https://arxiv.org/abs/2604.11035), with
full-parameter ternary QAT in VeOmni's text trainer.

The implementation was reviewed against paper sections 3.1–3.2 and appendices
E/H, plus the [released training and decoding code](https://github.com/Introspective-Diffusion/I-DLM/tree/a23c1a12ef997c7f3ad616b25bcfb62db39ded68).
The core adaptation keeps the paper's block-causal noisy/clean visibility,
one-position label shift, and acceptance with residual correction. Maple's
ternary QAT, sliding attention, and FSDP2 are architecture/framework adaptations;
the authors' reported Qwen3 results do not establish Maple quality or speed.

## Setup and preflight

Use `uv sync --frozen --extra gpu --dev --python 3.12` to install the pinned
environment. Reuse existing latent-master weights for native Maple QAT.
Allow 80 GiB free for rotation of two BF16 weight checkpoints, or 160 GiB with
FP32 master weights, plus the raw-text Arrow cache used by the native mapped
dataset. The launcher checks both and follows the configured master precision:

```bash
MAPLE_ROOT=outputs/maple_idlm_88k_online
MAPLE_MODEL=/path/to/v7_88k_maple_latentmaster
.venv/bin/python scripts/maple_idlm/prepare_data.py --root "$MAPLE_ROOT" \
  --model-path "$MAPLE_MODEL" --ternary-scheme row_twn --tokenization online
```

Online preparation pins the dataset revision, reserves the mask token, and
checks all raw Parquet shards without tokenizing the corpus. During training,
VeOmni's native mapped dataset and four workers per GPU tokenize conversations
as they are sampled. The launcher keeps the raw-text Arrow cache under
`$MAPLE_ROOT/dataset_cache` on persistent storage. No tokenized dataset is written.
Local model files are used in place and never modified. The
manifest fingerprints model/tokenizer metadata and weight-file sizes/mtimes;
launch rejects changes to that identity. Omitting `--model-path` downloads the
pinned public checkpoint and defaults to its `group_absmax` recipe. Preparation
holds the same lock as training. `--tokenization offline` remains available to
write tokenized shards and count their tokens; repeating it resumes completed
shards. Changing the mode, shard count, sequence length, or source
revision requires a new root so stale files cannot enter training.

## Eight-H100 throughput preset

`configs/text/maple_idlm_8gpu_fast.yaml` uses packed Quack experts with fused
activation kernels, native ternary QAT, fused AdamW with FP32 master weights,
BF16 compute, and FSDP2 across eight GPUs. Async activation offload allows
microbatch 20; its idle host cache is capped at 24 GiB per rank (192 GiB per node).
In-flight host buffers are additional.
Use the CUDA async allocator with this preset:

```bash
PYTORCH_ALLOC_CONF=backend:cudaMallocAsync ./start_maple.sh \
  --root "$MAPLE_ROOT" --config configs/text/maple_idlm_8gpu_fast.yaml \
  --gpus 8 --run-name maple-b1-fused-4b --hours 16 --prepare-only
```

The proposed schedule uses microbatch 20, global batch 160, and 6,104 updates
for a nominal 4B-token budget at length 4096. Padding makes the actual consumed
input count slightly smaller. It warms up from zero to `1e-5` over 100 updates,
then applies cosine decay to `1e-6`. Weights-only checkpoints are saved every
1,000 updates and at the final epoch boundary; the default retention keeps one
complete save. This FP32-master preset requires 160 GiB free for safe rotation.
It starts fresh from the prepared model, with no checkpoint resume.

Both presets set `train.checkpoint.save_optimizer` and `load_optimizer` to
`false`. Continuation restores weights and the schedule with a fresh optimizer;
enable both flags for full optimizer resume.

A 20-step benchmark measured 12,081 original input tokens/s/GPU and 18.6%
useful MFU over steps 5–20, a 16.8% gain over the batch-16 control. Sampled
peak device memory was 76.4 GiB. Four billion tokens take about 11h30m plus
startup and checkpoint saves at that rate. These short benchmarks used constant
LR `1e-5` without warmup and do not establish long-run convergence.

`--prepare-only` writes `resolved.yaml` without starting training or W&B.
Review it, run `.venv/bin/wandb login`, then repeat the command without
`--prepare-only` to start. Use a terminal multiplexer for a long run.

## Single-GPU run

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
220 GiB available host RAM. Weight checkpoints are about 38 GiB; the launcher
requires 80 GiB free after preparation for a fresh run with one retained save.
With optimizer saves enabled, a full checkpoint is about 151 GiB and rotation
requires 320 GiB free.
Training keeps one completed checkpoint per run; `MAPLE_KEEP_CHECKPOINTS`
overrides this. Older runs are untouched.

The launcher runs `torchrun` in the foreground with output in `training.log`.
`--prepare-only` writes the config without training. Short eight-H100 throughput
benchmarks are supported; establish convergence separately before a long run.
Set `OMP_NUM_THREADS` to override the default CPU thread limit.

To resume, set `train.checkpoint.load_path: auto` and supply the existing
`--run-name`. Its deadline survives restart; `--fresh-window` starts a new time
window. Fresh training uses `load_path: null` and rejects reused training state.
A resume must find a completed checkpoint with matching GPU count and all
rank-local cursors; a failed automatic lookup is an error. A lock prevents
concurrent launches against the same data root. Keep the same data-worker count
when restoring native dataloader state. W&B and
`metrics.jsonl` record training progress. At the deadline, training stops between
updates and saves weights and data cursors (plus optimizer state when enabled),
even between scheduled saves. Nonfinite loss or gradients abort before an update.

## Data, objective and kernels

Maple's MFU estimate counts the forward and backward matrix multiplications of
the attention projections, router, selected experts, and LM head. It counts
causal attention edges within each packed document, including the sliding
window, and both streams when I-DLM is enabled. Input tokens/s still counts
the original tokens before stream duplication. Activation recomputation, QAT,
optimizer work, and communication are excluded from model FLOPs; GPU activity
reported by `nvidia-smi` is a different utilization measure.

Preparation reuses the selected local weights or downloads pinned public Maple
weights, then downloads all 120 OpenThoughts3 shards and records provenance in
`manifest.json`. The same transform runs offline or inside training workers:
it applies Maple's chat template, supervises responses including reasoning,
and keeps the first 4096 tokens.
A prompt-hashed 1% validation split stays out of training. `--shards N --root PATH`
prepares a smaller experiment; preparation can restart. OpenThoughts is the
authors' public example dataset, not the paper's original corpus.

Training concatenates `[noisy | clean]` with duplicated logical positions. Noisy
queries see causal tokens in their own block and clean tokens in earlier blocks;
clean queries see only their causal clean prefix. Packed documents and padding
are isolated, and labels shift once. Training divides both branch loss sums by
the unshifted supervised-label count, matching the trainer's global token
weighting. This keeps fixed-weight loss and gradients independent of response
packing and rank partition. The logged branch CE values use their respective
valid target counts. Loss combines the masked term with
`idlm_clean_weight * clean_term`. The single-GPU example uses block size 1 and
clean weight 0.2, with ten warmup updates. This pilot does not reproduce the
paper's full curriculum: later stages use block sizes 2 and 3 and auto-balanced
loss. The matching decoding stride is block size + 1.

FlexAttention implements the training mask and backward. Its sparse metadata
is built from 128-token tile bounds, avoiding a dense quadratic token mask
during construction. Conservative partial tiles retain the exact token-level
mask; only tiles proven fully visible bypass it. Document boundaries, padding,
sliding windows, and the noisy/clean boundary remain isolated.
Grouped MoE GEMMs use
`model.ops_implementation.moe_implementation: fused_triton` or `fused_quack`.
Quack uses the existing CUTLASS/CuTe backend on SM90+ GPUs; both choices retain
the same ternary QAT, routing, and clamped SwiGLU semantics. Experts use two
packed Parameters per layer: `experts.gate_up_proj` has shape `[E, 2*I, H]`
with gate rows followed by up rows, and `experts.down_proj` is `[E, H, I]`.
Quack gathers token rows inside weight-gradient GEMMs and fuses clamp, SwiGLU,
and routing multiplication. Backward recomputes activation intermediates while
retaining low-precision rounding; the native TWN quantizer is unchanged.
Liger supplies linear cross entropy and RMSNorm. FSDP2 data sharding is
supported; SP/EP sizes greater than one are
rejected. Maple's QK normalization, partial RoPE and configured global-attention
RoPE behavior are preserved. The local 88k checkpoint uses RoPE on global
attention (`nope_on_global_attention: false`).

Decoder layers use Transformers' `GradientCheckpointingLayer` for recomputation
and async activation offload. The fast preset enables offload; the single-GPU
preset leaves it disabled.

`expert_weight_layout: packed_gate_up` records this model layout in saved
configuration. VeOmni's runtime loader converts the original per-expert Maple
HF weights into packed tensors as shards are read; it does not rewrite or
download the source checkpoint. It rejects missing, duplicate, mixed-layout,
or incorrectly shaped tensors. Packed HF saves pass through without conversion.
The HF export index maps to the packed names so expert weights are retained.
Use the registered VeOmni Maple model to load these exports.

New DCP checkpoints contain packed keys. An older per-expert DCP checkpoint
must first be exported with `scripts/merge_dcp_to_hf.py` (the command below), then
loaded as HF weights into a fresh packed run. This migrates weights only;
optimizer state and data cursors do not transfer. Use `load_path: null` and
`idlm_initialize_mask_token: false` to preserve the trained mask-token rows.
Direct legacy DCP resume is rejected because its parameter keys differ.

Native latent masters use `ternary_scheme: row_twn`: select absolute weights
above `0.7 * mean(abs(weight))` per output row and scale signs by the mean
absolute selected weight. Floating-point reductions remain native PyTorch;
changing their order can flip ternary states. The public checkpoint uses
`group_absmax` over 128 input features with a Triton quantizer. Both use identity
STE and BF16 GEMMs; routers, embeddings, norms, and the language head stay
floating point. Use local latent masters when available.

The new `<|idlm_mask|>` token uses an unused padded vocabulary row. Fresh training
initializes its input/output rows with the vocabulary mean plus seeded noise;
resume preserves them. Preparation writes the registered Maple config without
executing downloaded model code.

## Export and evaluate

Replace `RUN` and `STEP` with an actual run directory and completed step:

```bash
.venv/bin/python scripts/merge_dcp_to_hf.py \
  --load-dir "$MAPLE_ROOT/RUN/checkpoints/global_step_STEP" \
  --save-dir "$MAPLE_ROOT/export" \
  --model-assets-dir "$MAPLE_ROOT/RUN/model_assets"
.venv/bin/torchrun --standalone --nproc-per-node=1 scripts/maple_idlm/generate.py \
  "$MAPLE_ROOT/export" --stride 2 --tokens 128
.venv/bin/torchrun --standalone --nproc-per-node=1 scripts/maple_idlm/evaluate.py \
  "$MAPLE_ROOT/export" --data-root "$MAPLE_ROOT" --output "$MAPLE_ROOT/validation.json"
```

Exports contain floating QAT masters plus configuration and tokenizer assets.
Use the provided decoder so the same ternary quantization applies on every
forward. When starting a new training run from an export, set
`idlm_initialize_mask_token: false` to preserve its trained mask rows.

The decoder uses exact `p/q` acceptance with residual correction, causal/sliding
SDPA and KV rollback; `--no-cache` provides a reference. It preserves the
converted model's AR distribution, not necessarily the original Maple model's
quality. It does not implement the paper's serving system or claim its throughput.

Validation samples up to 64 held-out examples and reports both losses. For an
online root, it selects raw held-out conversations and tokenizes only that
sample; examples with no usable response after truncation are excluded.
Maple MFU uses the analytical model-FLOPs accounting described above; it is
available in current training logs. Older logs written before the estimator was
registered contain zero and cannot be treated as utilization measurements.

Run the focused tests with:

```bash
.venv/bin/pytest -q tests/models/test_maple_idlm.py
```

The focused model suite is selected by GPU CI. Preparation and checkpoint
regressions live in `tests/data/test_maple_preparation.py` and
`tests/checkpoints/test_maple_launch.py`, both selected by the directory-level
GPU/NPU jobs. Maple's accelerator kernels currently target CUDA.

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
