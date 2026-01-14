from peft.utils import register_peft_method

from .config import LoraMoEConfig
from .layer import LoraMoELinear
from .model import LoraMoEModel


__all__ = ["LoraMoEConfig", "LoraMoELinear", "LoraMoEModel"]


register_peft_method(
    name="loramoe", config_cls=LoraMoEConfig, model_cls=LoraMoEModel, prefix="lora_", is_mixed_compatible=True
)


def __getattr__(name):

    raise AttributeError(f"module {__name__} has no attribute {name}")