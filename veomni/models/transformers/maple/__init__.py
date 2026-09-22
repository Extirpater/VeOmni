from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("maple")
def register_maple_config():
    from .configuration_maple import MapleConfig

    return MapleConfig


@MODELING_REGISTRY.register("maple")
def register_maple_model(architecture):
    from .checkpoint_tensor_converter import (
        convert_maple_fqn_to_index_mapping,
        create_maple_checkpoint_tensor_converter,
    )
    from .generated.patched_modeling_maple_gpu import MapleForCausalLM, MapleModel

    for model_cls in (MapleForCausalLM, MapleModel):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_maple_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_maple_fqn_to_index_mapping)
    return MapleModel if architecture == "MapleModel" else MapleForCausalLM
