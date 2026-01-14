import warnings
from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType

@dataclass
class LoraMoEConfig(LoraConfig):  # TODO: Inherit LoraConfig to reuse basic parameters
    """
    Configuration class for LoRA-MoE.
    """
    peft_type: Optional[PeftType] = field(default=PeftType.LORAMOE, metadata={"help": "The type of PEFT model."})
    num_experts: int = field(default=4, metadata={"help": "Number of LoRA experts."})
    top_k: int = field(default=2, metadata={"help": "Number of top experts to route the input to."})

    def __post_init__(self):
        super().__post_init__()  # Call the parent class's __post_init__ for basic validation
        self.peft_type = PeftType.LORAMOE
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) must be less than or equal to num_experts ({self.num_experts})")