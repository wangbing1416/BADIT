# src/peft/tuners/lora/lora_moe_layer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from peft.tuners.lora.layer import LoraLayer  # 假设 PEFT 有 LoraLayer 基类
from peft.tuners.lora import LoraMoEConfig  # 导入你的配置


class LoraMoELinear(nn.Module, LoraLayer):  # 继承 LoraLayer 以复用部分逻辑
    def __init__(self, base_layer: nn.Linear, config: LoraMoEConfig, adapter_name: str):
        super().__init__()
        LoraLayer.__init__(self, base_layer, adapter_name) # 初始化 LoraLayer 基类

        self.base_layer = base_layer
        # 从 config 继承 LoRA 参数
        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.lora_dropout = config.lora_dropout
        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
        # LoRA-MoE 特定参数
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        # --- LoRA Experts ---
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.empty(self.r, base_layer.in_features)) for _ in range(self.num_experts)]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.empty(base_layer.out_features, self.r)) for _ in range(self.num_experts)]
        )

        # --- Gate Network ---
        self.gate = nn.Linear(base_layer.in_features, self.num_experts, bias=False)

        # --- Dropout ---
        if self.lora_dropout > 0:
            self.lora_dropout_layer = nn.Dropout(p=self.lora_dropout)
        else:
            self.lora_dropout_layer = lambda x: x

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化 LoRA A/B 矩阵
        for i in range(self.num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])
        # 初始化 Gate 权重 (可选)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Base Forward ---
        base_output = self.base_layer(x)

        # --- Gate & Routing ---
        gate_input = x.mean(dim=-2) if x.dim() > 2 else x
        gate_logits = self.gate(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # --- Compute LoRA Contributions ---
        batch_size = x.size(0)
        seq_len = x.size(1) if x.dim() > 2 else 1

        top_k_flat_indices = top_k_indices.view(-1)
        selected_A_stacked = torch.stack([self.lora_A[idx] for idx in top_k_flat_indices], dim=0)
        selected_B_stacked = torch.stack([self.lora_B[idx] for idx in top_k_flat_indices], dim=0)

        x_expanded = x.unsqueeze(1).expand(-1, self.top_k, -1, -1).contiguous().view(batch_size * self.top_k, seq_len, -1)
        lora_delta_intermediate = torch.bmm(x_expanded, selected_A_stacked.transpose(-1, -2))
        lora_delta_full = torch.bmm(lora_delta_intermediate, selected_B_stacked.transpose(-1, -2))
        lora_delta = lora_delta_full.view(batch_size, self.top_k, seq_len, -1)

        weights_expanded = top_k_weights.unsqueeze(-1).unsqueeze(-1)
        lora_delta_weighted = lora_delta * weights_expanded * self.scaling
        lora_output = lora_delta_weighted.sum(dim=1)

        # --- Combine ---
        final_output = base_output + lora_output
        return final_output

    # 可能需要实现 update_layer, merge, unmerge 等方法，根据 PEFT 框架要求
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        # 实现参数更新逻辑
        pass

# 可以为其他层类型（如 Conv2d）实现 LoraMoEConv2d 等
# class LoraMoEConv2d(nn.Module, LoraLayer): ...