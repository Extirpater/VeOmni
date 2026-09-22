"""Strict streaming conversion of Maple expert weights to concatenated packed storage."""

import re

from ..._moe_fused_weight_map import PER_EXPERT_SPLIT_TO_FUSED_PATTERN, convert_per_expert_fqn_mapping_to_fused
from ...checkpoint_tensor_loading import ConvertedCheckpointTensor


_PACKED_PATTERN = re.compile(r"^(.+\.mlp)\.experts\.(gate_up_proj|down_proj)$")


class MapleCheckpointTensorConverter:
    def __init__(
        self,
        num_experts,
        hidden_size,
        intermediate_size,
        *,
        expected_shapes=None,
        buffer_shapes=None,
        optional_names=(),
    ):
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expected_shapes = expected_shapes
        self.buffer_shapes = buffer_shapes or {}
        self.optional_names = set(optional_names)
        self._seen_inputs = set()
        self._emitted = set()
        self._layouts = {}
        self._buffers = {}
        self._parts = {}

    def can_handle(self, name):
        return self.expected_shapes is not None or bool(
            PER_EXPERT_SPLIT_TO_FUSED_PATTERN.match(name) or _PACKED_PATTERN.match(name)
        )

    def _set_layout(self, prefix, layout):
        previous = self._layouts.setdefault(prefix, layout)
        if previous != layout:
            raise ValueError(f"Mixed packed and per-expert Maple weights for {prefix}")

    def _emit(self, name, tensor):
        if name in self._emitted:
            raise ValueError(f"Duplicate Maple target tensor: {name}")
        if self.expected_shapes is not None:
            expected = self.expected_shapes.get(name, self.buffer_shapes.get(name))
            if expected is None:
                raise ValueError(f"Unexpected Maple checkpoint tensor: {name}")
            if tuple(tensor.shape) != tuple(expected):
                raise ValueError(f"Incorrect Maple shape for {name}: {tuple(tensor.shape)} != {tuple(expected)}")
        self._emitted.add(name)
        return ConvertedCheckpointTensor(name, tensor)

    def convert(self, name, tensor):
        match = PER_EXPERT_SPLIT_TO_FUSED_PATTERN.match(name)
        # Normalize expert IDs so aliases such as "0" and "00" are duplicates.
        source_key = (match[1], int(match[2]), match[3]) if match else name
        if source_key in self._seen_inputs:
            raise ValueError(f"Duplicate Maple checkpoint tensor: {name}")
        self._seen_inputs.add(source_key)
        if match is None:
            packed = _PACKED_PATTERN.match(name)
            if packed is not None:
                prefix, projection = packed.groups()
                self._set_layout(prefix, "packed")
                shape = (
                    (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
                    if projection == "gate_up_proj"
                    else (self.num_experts, self.hidden_size, self.intermediate_size)
                )
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"Incorrect Maple packed shape for {name}: {tuple(tensor.shape)} != {shape}")
            return self._emit(name, tensor)

        prefix, expert_id, projection = source_key
        self._set_layout(prefix, "per_expert")
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(f"Maple expert index out of range: {name}")
        down = projection == "down_proj"
        shape = (self.hidden_size, self.intermediate_size) if down else (self.intermediate_size, self.hidden_size)
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Incorrect Maple expert shape for {name}: {tuple(tensor.shape)} != {shape}")
        target = f"{prefix}.experts.{'down_proj' if down else 'gate_up_proj'}"
        if self.expected_shapes is not None and target not in self.expected_shapes:
            raise ValueError(f"Unexpected Maple expert layer: {name}")
        if target not in self._buffers:
            packed_shape = (
                (self.num_experts, self.hidden_size, self.intermediate_size)
                if down
                else (self.num_experts, 2 * self.intermediate_size, self.hidden_size)
            )
            self._buffers[target] = tensor.new_empty(packed_shape)
            self._parts[target] = 0
        buffer = self._buffers[target]
        if buffer.dtype != tensor.dtype or buffer.device != tensor.device:
            raise ValueError(f"Inconsistent Maple expert dtype/device for {target}")
        if down:
            buffer[expert_id].copy_(tensor)
        else:
            offset = 0 if projection == "gate_proj" else self.intermediate_size
            buffer[expert_id, offset : offset + self.intermediate_size].copy_(tensor)
        self._parts[target] += 1
        if self._parts[target] != self.num_experts * (1 if down else 2):
            return None
        del self._parts[target]
        return self._emit(target, self._buffers.pop(target))

    def finalize(self):
        if self._buffers:
            raise RuntimeError(f"Incomplete Maple expert tensors: {sorted(self._buffers)}")
        if self.expected_shapes is not None:
            missing = self.expected_shapes.keys() - self._emitted - self.optional_names
            if missing:
                raise RuntimeError(f"Missing Maple checkpoint tensors: {sorted(missing)}")
        return []


def create_maple_checkpoint_tensor_converter(model):
    config = model.config
    # to_empty() can break this alias before the loader restores tied weights.
    # HF saves the embedding once; require its canonical key, not a second copy.
    optional_names = ("lm_head.weight",) if config.tie_word_embeddings else ()
    return MapleCheckpointTensorConverter(
        config.num_experts,
        config.hidden_size,
        config.moe_intermediate_size,
        expected_shapes={name: tuple(param.shape) for name, param in model.named_parameters(remove_duplicate=False)},
        buffer_shapes={name: tuple(buffer.shape) for name, buffer in model.named_buffers()},
        optional_names=optional_names,
    )


convert_maple_fqn_to_index_mapping = convert_per_expert_fqn_mapping_to_fused
