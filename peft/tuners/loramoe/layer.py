# src/peft/tuners/lora/lora_moe_layer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from peft.tuners.lora.layer import LoraLayer  # Import LoraLayer base class
from peft.tuners.loramoe.config import LoraMoEConfig  # Import your config
from transformers.utils import transpose
from sklearn.cluster import KMeans
import numpy as np


class LoraMoELinear(nn.Module, LoraLayer):  # Inherits nn.Module and LoraLayer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_lora_weights: bool | str = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        # --- LoraMoE specific parameters ---
        num_experts: int = 4,
        top_k: int = 2,
        # ... other possible parameters ...
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer, adapter_name)  # Initialize LoraLayer base class

        # --- LoraLayer basic attributes ---
        self.fan_in_fan_out = fan_in_fan_out
        self._active_adapter = adapter_name
        # ... possibly other LoraLayer basic attributes ...

        # --- LoRA parameters (need to manage multiple adapters) ---
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        # ... possibly other LoRA parameters ...

        # --- LoRA-MoE specific parameters ---
        self.num_experts = num_experts
        self.top_k = top_k

        # --- Store LoRA-MoE parameters for multiple adapters ---
        self.lora_A: Dict[str, nn.ParameterList] = nn.ParameterDict()
        self.lora_B: Dict[str, nn.ParameterList] = nn.ParameterDict()
        self.lora_dropout: Dict[str, nn.Module] = nn.ModuleDict()
        self.scaling: Dict[str, float] = {}
        # self.lora_expert_names: Dict[str, list] = {} # If you need to store expert names

        # --- Gate Network (per-adapter) ---
        self.gate: Dict[str, nn.Linear] = nn.ModuleDict()

        # Initialize parameters for the current adapter
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora)
        self._cached_grads = None  # Used to cache gradients: List[Tuple[torch.Tensor, torch.Tensor]] for each expert

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        init_lora_weights: bool | str,
        use_rslora: bool = False,
        use_dora: bool = False,
        # ... other parameters ...
    ):
        """Update or initialize the LoRA-MoE parameters for the specified adapter_name."""
        # Create LoRA expert parameters for the new adapter
        lora_A_adapters = nn.ParameterList(
            [nn.Parameter(torch.empty(r, self.in_features)) for _ in range(self.num_experts)]
        )
        lora_B_adapters = nn.ParameterList(
            [nn.Parameter(torch.empty(self.out_features, r)) for _ in range(self.num_experts)]
        )

        # After creating lora_A_adapters / lora_B_adapters in update_layer
        self._last_grads_cache = {}  # {param_id: grad_clone}

        def make_grad_hook(param_name):
            def hook(grad):
                # Save a copy of the gradient (note: grad is read-only, cannot be modified)
                self._last_grads_cache[param_name] = grad.clone().detach()
                return None  # Do not modify the gradient

            return hook

        # Register hook for each expert's A/B parameters
        for e in range(self.num_experts):
            param_A = lora_A_adapters[e]
            param_B = lora_B_adapters[e]
            param_A.register_hook(make_grad_hook(f"A_{e}"))
            param_B.register_hook(make_grad_hook(f"B_{e}"))

        # Create Gate network for the new adapter
        gate_adapter = nn.Linear(self.in_features, self.num_experts, bias=False)

        # Store
        self.lora_A[adapter_name] = lora_A_adapters
        self.lora_B[adapter_name] = lora_B_adapters
        self.gate[adapter_name] = gate_adapter
        self.scaling[adapter_name] = lora_alpha / r if r > 0 else 1.0

        # Initialize parameters
        if init_lora_weights == "gaussian":
            for i in range(self.num_experts):
                nn.init.normal_(lora_A_adapters[i], std=1 / r)
                nn.init.normal_(lora_B_adapters[i], std=1 / r)
        elif init_lora_weights == "pissa_moe":
            self.pissa_moe_init(adapter_name)
        elif init_lora_weights:
            for i in range(self.num_experts):
                nn.init.kaiming_uniform_(lora_A_adapters[i], a=math.sqrt(5))
                nn.init.zeros_(lora_B_adapters[i])
        else:
            for i in range(self.num_experts):
                nn.init.zeros_(lora_A_adapters[i])
                nn.init.zeros_(lora_B_adapters[i])

        # Initialize gate (optional)
        # nn.init.xavier_uniform_(gate_adapter.weight)

        # Dropout
        if lora_dropout > 0.0:
            lora_dropout_adapter = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_adapter = nn.Identity()
        self.lora_dropout[adapter_name] = lora_dropout_adapter

        # Set to training mode
        self.set_adapter(self.active_adapters + [adapter_name])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # --- Get current active adapter ---
        # First try to get it from kwargs; this is set during mixed-adapter inference
        adapter_names = kwargs.get("adapter_names", None)
        # If adapter_names is not in kwargs, use instance variable _active_adapter
        # Note: _active_adapter is usually a string, but in some PEFT internal logic it can also be a list
        active_adapter = self._active_adapter

        # Handle the case where adapter_names is a list (mixed-adapter inference)
        if adapter_names is not None:
            if len(adapter_names) != 1:
                # If multiple adapters need to be mixed, corresponding logic must be implemented
                # Currently, LoraMoE usually uses only one adapter's parameters in a single forward pass
                # Here we can simply take the first, or raise NotImplementedError
                # raise NotImplementedError("LoraMoE forward with multiple adapter_names not implemented.")
                # For simplicity, we take the first, but this may not be ideal
                # Alternatively, you can implement more complex logic to handle multiple experts or gates
                # This depends on whether your LoraMoE design supports using multiple adapters in a single forward pass
                # For standard single-adapter forward, adapter_names is usually also a single-element list
                active_adapter = adapter_names[0] if adapter_names else active_adapter
            else:
                active_adapter = adapter_names[0]  # Use the first adapter in the list
        else:
            # If adapter_names is not in kwargs, check if _active_adapter is a list
            # If it is a list, also take the first (this is uncommon in standard training/single-adapter inference,
            # but check for robustness)
            if isinstance(active_adapter, list):
                if active_adapter:
                    active_adapter = active_adapter[0]
                else:
                    # If the list is empty, use a default value or raise an error
                    active_adapter = "default"  # Or raise ValueError

        # Ensure active_adapter is a string
        if not isinstance(active_adapter, str):
            raise ValueError(f"active_adapter must be a string, got {type(active_adapter)}: {active_adapter}")

        # --- Base Forward ---
        base_output = self.base_layer(x)

        # --- Gate & Routing (for active adapter) ---
        gate_input = x.mean(dim=-2) if x.dim() > 2 else x
        gate_logits = self.gate[active_adapter](gate_input)  # Now active_adapter should be a string
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # --- Compute LoRA Contributions (for active adapter) ---
        batch_size = x.size(0)
        seq_len = x.size(1) if x.dim() > 2 else 1

        top_k_flat_indices = top_k_indices.view(-1)
        selected_A_stacked = torch.stack([self.lora_A[active_adapter][idx] for idx in top_k_flat_indices], dim=0)
        selected_B_stacked = torch.stack([self.lora_B[active_adapter][idx] for idx in top_k_flat_indices], dim=0)

        x_expanded = x.unsqueeze(1).expand(-1, self.top_k, -1, -1).contiguous().view(batch_size * self.top_k, seq_len, -1)
        lora_delta_intermediate = torch.bmm(x_expanded, selected_A_stacked.transpose(-1, -2))
        lora_delta_full = torch.bmm(lora_delta_intermediate, selected_B_stacked.transpose(-1, -2))
        lora_delta = lora_delta_full.view(batch_size, self.top_k, seq_len, -1)

        weights_expanded = top_k_weights.unsqueeze(-1).unsqueeze(-1)
        lora_delta_weighted = lora_delta * weights_expanded * self.scaling[active_adapter]
        lora_output = lora_delta_weighted.sum(dim=1)

        # --- Combine ---
        final_output = base_output + lora_output
        return final_output

    def pissa_moe_init(self, adapter_name: str):
        """
        Initialize LoRA-MoE using SVD decomposition of the base weight.
        Decomposes W into num_experts low-rank matrices (each of rank r) that are orthogonal in SVD space.
        """
        weight = self.get_base_layer().weight  # shape: [out_features, in_features]
        dtype = weight.dtype

        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA-MoE under float32, float16, or bfloat16."
            )

        # Handle fan_in_fan_out (e.g., for Conv1D in GPT2)
        # W = transpose(weight.to(torch.float32), self.fan_in_fan_out)  # [out, in]
        weight_f32 = weight.to(torch.float32)
        W = weight_f32.T if self.fan_in_fan_out else weight_f32

        r = self.r
        num_experts = self.num_experts
        total_rank = r * num_experts

        if total_rank > min(W.shape):
            raise ValueError(
                f"Total rank (num_experts * r = {total_rank}) exceeds matrix rank limit {min(W.shape)}. "
                "Reduce num_experts or r."
            )

        # Perform SVD
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U: [out, k], S: [k], Vh: [k, in]
        k = S.size(0)

        if total_rank > k:
            total_rank = k  # fallback: use all available components
            # Adjust r or num_experts? For simplicity, we keep num_experts and reduce r per expert
            r_actual = total_rank // num_experts
            remainder = total_rank % num_experts
            # For simplicity, assume divisible; otherwise handle uneven split
            if remainder != 0:
                raise NotImplementedError(
                    "Uneven SVD split not implemented. Ensure num_experts * r <= min(in, out) and divisible.")
            r_use = r_actual
        else:
            r_use = r

        # Rescale singular values by scaling factor (like PiSSA)
        scaling = self.scaling[adapter_name]
        sqrt_S = torch.sqrt(S[:total_rank] / scaling)  # shape: [total_rank]

        # Split U, sqrt_S, Vh into num_experts chunks of size r_use
        U_chunks = U[:, :total_rank].split(r_use, dim=1)  # list of [out, r]
        Vh_chunks = Vh[:total_rank, :].split(r_use, dim=0)  # list of [r, in]
        sqrt_S_chunks = sqrt_S.split(r_use)  # list of [r]

        # Initialize each expert
        with torch.no_grad():
            for e in range(num_experts):
                # lora_B: [out, r], lora_A: [r, in]
                lora_B_e = U_chunks[e] * sqrt_S_chunks[e].unsqueeze(0)  # broadcast over out_dim
                lora_A_e = sqrt_S_chunks[e].unsqueeze(1) * Vh_chunks[e]  # broadcast over in_dim

                self.lora_B[adapter_name][e].copy_(lora_B_e)
                self.lora_A[adapter_name][e].copy_(lora_A_e)

        # Compute residual weight: W - Σ (lora_B[e] @ lora_A[e]) * scaling
        reconstructed = torch.zeros_like(W)
        for e in range(num_experts):
            delta_e = self.lora_B[adapter_name][e] @ self.lora_A[adapter_name][e]
            reconstructed += delta_e * scaling

        W_residual = W - reconstructed
        if self.fan_in_fan_out:
            W_residual = W_residual.T
        self.get_base_layer().weight.data.copy_(W_residual.to(dtype))

    def get_last_computed_gradients(self):
        """Return the gradients from the last backward computation (as a dictionary)"""
        return getattr(self, '_last_grads_cache', {})

    def regroup_by_cached_gradients(self, grad_cache: dict, adapter_name: str, _group_logger, module_name,
                                    current_epoch: int = None,
                                    max_epochs: float = None):
        """
        Regroup the LoRA-MoE basic units using cached gradients.
        Args:
            grad_cache (dict): {"A_0": grad, "B_0": grad, ...}
            adapter_name (str): current adapter name
            _group_logger: logger
            current_epoch (int, optional): current epoch (can start from 0 or 1)
            max_epochs (int, optional): total number of epochs
        """
        # === Condition 1: Skip regrouping after more than half the epochs ===
        if current_epoch is not None and max_epochs is not None:
            if current_epoch > max_epochs // 2:
                _group_logger.info(f"Epoch {current_epoch} > {max_epochs}//2, skip regrouping.")
                return

        if adapter_name not in self.lora_A or adapter_name not in self.lora_B:
            _group_logger.warning(f"Adapter {adapter_name} not found in lora_A/lora_B.")
            return

        K = self.num_experts
        r = self.r
        if r <= 0:
            _group_logger.warning("Rank r <= 0, skip regrouping.")
            return

        lora_A_list = self.lora_A[adapter_name]  # List[nn.Parameter], [r, in]
        lora_B_list = self.lora_B[adapter_name]  # List[nn.Parameter], [out, r]

        # === Step 1: Collect gradient vectors of all base units (convert to float32 to avoid BFloat16 errors) ===
        grad_vectors = []
        param_info = []  # (expert_idx, rank_idx)

        for e in range(K):
            gA = grad_cache.get(f"A_{e}")
            gB = grad_cache.get(f"B_{e}")
            if gA is None or gB is None:
                _group_logger.warning(f"Missing gradient for expert {e}. Skipping regroup.")
                return
            # Convert to float32 to avoid KMeans failure caused by BFloat16
            gA = gA.to(torch.float32)
            gB = gB.to(torch.float32)
            if gA.shape != (r, self.in_features) or gB.shape != (self.out_features, r):
                _group_logger.error(
                    f"Gradient shape mismatch for expert {e}. Expected A: ({r}, {self.in_features}), B: ({self.out_features}, {r})")
                return
            for i in range(r):
                vec = torch.cat([gA[i].flatten(), gB[:, i].flatten()])  # [in + out]
                grad_vectors.append(vec)
                param_info.append((e, i))

        if len(grad_vectors) != K * r:
            _group_logger.error("Gradient count mismatch.")
            return

        grad_matrix = torch.stack(grad_vectors)  # [K*r, D], dtype=float32
        norms = torch.norm(grad_matrix, dim=1, keepdim=True)
        eps = 1e-12
        norms = torch.where(norms < eps, torch.ones_like(norms), norms)
        grad_dirs = grad_matrix / norms  # Unit direction vectors [K*r, D]

        # === Step 2: KMeans clustering (spherical clustering) ===
        try:
            grad_dirs_np = grad_dirs.cpu().numpy()  # float32 → OK for sklearn
            kmeans = KMeans(n_clusters=K, random_state=0, n_init=10, max_iter=100)
            labels = kmeans.fit_predict(grad_dirs_np)
        except Exception as e:
            _group_logger.warning(f"KMeans failed: {e}. Falling back to random assignment.")
            labels = np.random.randint(0, K, size=K * r)

        # === Step 3: Greedy reassignment to ensure exactly r per cluster ===
        cluster_to_indices = {k: [] for k in range(K)}
        for idx, label in enumerate(labels):
            cluster_to_indices[label].append(idx)

        all_indices = list(range(K * r))
        np.random.shuffle(all_indices)
        clusters = []

        for k in range(K):
            current = cluster_to_indices[k]
            if len(current) < r:
                remaining = [i for i in all_indices if i not in current and not any(i in c for c in clusters)]
                if len(remaining) < r - len(current):
                    remaining = all_indices  # fallback
                current.extend(np.random.choice(remaining, r - len(current), replace=False))
            elif len(current) > r:
                current = np.random.choice(current, r, replace=False).tolist()
            clusters.append(current[:r])

        # === Step 4: Regroup parameters (keep original dtype, e.g., bfloat16) ===
        with torch.no_grad():
            all_A_vals = []
            all_B_vals = []
            for (e, i) in param_info:
                # Keep original dtype (e.g., bfloat16)
                all_A_vals.append(lora_A_list[e].data[i].clone())
                all_B_vals.append(lora_B_list[e].data[:, i].clone())

            for new_e in range(K):
                new_A_rows = []
                new_B_cols = []
                for idx in clusters[new_e]:
                    idx = idx % len(all_A_vals)
                    new_A_rows.append(all_A_vals[idx])
                    new_B_cols.append(all_B_vals[idx])
                new_A = torch.stack(new_A_rows, dim=0)  # [r, in]
                new_B = torch.stack(new_B_cols, dim=1)  # [out, r]
                lora_A_list[new_e].copy_(new_A)
                lora_B_list[new_e].copy_(new_B)

        # === Step 2: Do not cluster; group directly in original order ===
        # Assume grad_dirs has shape [K*r, D]; group directly in order
        # original_indices = list(range(K * r))
        # clusters = []
        # for k in range(K):
        #     start_idx = k * r
        #     end_idx = (k + 1) * r
        #     cluster = original_indices[start_idx:end_idx]
        #     clusters.append(cluster)

        # === Step 5: Compute and log intra / inter cosine similarity ===
        with torch.no_grad():
            intra_cos_list = []
            expert_centroids = []
            eps = 1e-12

            for cluster in clusters:
                if len(cluster) < 2:
                    intra_cos_list.append(1.0)
                    vecs = grad_dirs[cluster]
                    centroid = vecs.mean(dim=0)
                    centroid = centroid / (centroid.norm() + eps)
                    expert_centroids.append(centroid)
                    continue

                vecs = grad_dirs[cluster]  # [r, D]
                normed = vecs / (vecs.norm(dim=1, keepdim=True) + eps)  # [r, D]
                cos_mat = torch.mm(normed, normed.T)  # [r, r]
                mask = torch.triu(torch.ones_like(cos_mat), diagonal=1).bool()
                if mask.any():
                    intra_cos = cos_mat[mask].mean().item()
                else:
                    intra_cos = 1.0
                intra_cos_list.append(intra_cos)

                centroid = vecs.mean(dim=0)
                centroid = centroid / (centroid.norm() + eps)
                expert_centroids.append(centroid)

            intra_mean = float(np.mean(intra_cos_list))

            if len(expert_centroids) > 1:
                centroids = torch.stack(expert_centroids)  # [K, D]
                inter_cos_mat = torch.mm(centroids, centroids.T)  # [K, K]
                mask = ~torch.eye(len(centroids), dtype=torch.bool, device=centroids.device)
                inter_mean = inter_cos_mat[mask].mean().item()
            else:
                inter_mean = 0.0

        _group_logger.info(
            f"✅ Regrouped adapter '{adapter_name}' with {K} experts (rank={r}). module name: {module_name}."
            f"Intra-expert cos: {intra_mean:.4f}, Inter-expert cos: {inter_mean:.4f}"
        )


def dispatch_loramoe(target: nn.Module, adapter_name: str, lora_config: LoraMoEConfig, **kwargs) -> Optional[nn.Module]:
    """
    Dispatch function to create a LoraMoE layer based on the target module type.
    """
    new_module = None

    # Check target module type and config type
    # Extract parameters from kwargs
    r = kwargs.pop("r", 0)
    lora_alpha = kwargs.pop("lora_alpha", 1)
    lora_dropout = kwargs.pop("lora_dropout", 0.0)
    fan_in_fan_out = kwargs.pop("fan_in_fan_out", False)
    init_lora_weights = kwargs.pop("init_lora_weights", True)
    use_rslora = kwargs.pop("use_rslora", False)
    use_dora = kwargs.pop("use_dora", False)
    # ... extract other possible parameters ...

    # Create LoraMoELinear instance
    new_module = LoraMoELinear(
        base_layer=target,
        adapter_name=adapter_name,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        fan_in_fan_out=fan_in_fan_out,
        init_lora_weights=init_lora_weights,
        use_rslora=use_rslora,
        use_dora=use_dora,
        # --- LoraMoE specific parameters ---
        num_experts=lora_config.num_experts,
        top_k=lora_config.top_k,
        # ... other LoraMoE specific parameters ...
    )

    return new_module
