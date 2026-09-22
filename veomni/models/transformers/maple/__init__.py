from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("maple")
def register_maple_config():
    from .configuration_maple import MapleConfig

    return MapleConfig


@MODELING_REGISTRY.register("maple")
def register_maple_model(architecture):
    from .generated.patched_modeling_maple_gpu import MapleForCausalLM, MapleModel

    return MapleModel if architecture == "MapleModel" else MapleForCausalLM
