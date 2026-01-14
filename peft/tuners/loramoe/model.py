# src/peft/tuners/lora/lora_moe_model.py

from typing import Optional
from torch import nn
# from peft.peft_model import PeftModelForCausalLM # assume you mainly handle Causal LM
# For other tasks, you may need PeftModelForSeq2SeqLM, PeftModelForSequenceClassification, etc.
# from peft.peft_model import PeftModel # or directly inherit the base class PeftModel
from peft.tuners.loramoe.config import LoraMoEConfig
from peft.tuners.lora import LoraModel  # import LoraModel base class
from peft.utils import _get_submodules
from peft.utils.other import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING  # may be needed as a base mapping

# If your LoraMoE layer implements the LoraLayer interface, this import is important
from .layer import LoraMoELinear, dispatch_loramoe  # import your implemented layers


class LoraMoEModel(LoraModel):  # inherit from LoraModel
    """
    A PEFT model that applies LoRA-MoE to a base transformer model.
    Inherits from LoraModel and overrides _create_new_module to inject LoraMoE logic.
    """
    # Optionally override prefix or other class attributes
    # prefix: str = "loramoe_" # e.g., if a different prefix is needed

    def __init__(self, model: nn.Module, peft_config, adapter_name: str = "default"):
        # Call the parent LoraModel initialization
        # This triggers BaseTuner.__init__, which calls inject_adapter
        # inject_adapter uses LoraMoEModel's _create_new_module method
        if isinstance(peft_config, dict):
            peft_config = peft_config[adapter_name]
        super().__init__(model, peft_config, adapter_name)
        # LoraModel __init__ already finished the core module replacement
        # because we overrode _create_new_module

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        """
        Override the _create_new_module method from LoraModel.
        This method defines how LoraMoE layers are created.
        """
        # Collect dispatcher functions. Add LoraMoE dispatcher first.
        dispatchers = []

        # Add your LoraMoE dispatcher *first* to handle LoraMoEConfig
        dispatchers.append(dispatch_loramoe)

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported for LoraMoE. "
                "Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`."
            )
        return new_module

