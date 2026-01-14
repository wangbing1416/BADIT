import torch
import os
from transformers import Seq2SeqTrainer, Trainer
from typing import Dict, List, Union
import numpy as np
import pickle
import joblib
import torch.distributed as dist
from torch.distributed import get_rank


def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    # If predictions_ids is a list of lists
    if isinstance(predictions_ids, list):
        # First, pad to the same length
        max_len = max(len(x) for x in predictions_ids)
        padded = []
        for seq in predictions_ids:
            padded_seq = [token_id if token_id != ignore_idx else tokenizer.pad_token_id for token_id in seq]
            padded_seq = padded_seq + [tokenizer.pad_token_id] * (max_len - len(padded_seq))
            padded.append(padded_seq)
        predictions_ids = np.array(padded)
    else:
        # Already a numpy array, process directly
        predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    ANSWER_PREFIX = "Output:"
    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    for pred in predictions:
        if tokenizer.bos_token:  # qwen3 has no tokenizer.bos_token
            pred = pred.replace(tokenizer.bos_token, '')
        pred = pred.replace(tokenizer.eos_token, '')
        pred = pred.replace(tokenizer.pad_token, '')
        if ANSWER_PREFIX in pred:
            splits = pred.split(ANSWER_PREFIX)
            final_predictions.append(splits[-1].strip())
        else:
            final_predictions.append(pred.strip())

    return final_predictions


class QKVParameterGradientTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        *args,
        qkv_gradient_save_path="./qkv_param_grads_avg.pt",
        mlp_gradient_save_path="./mlp_param_grads_avg.pt",
        mlp_activation_save_path="./mlp_param_activation.pt",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.qkv_gradient_save_path = qkv_gradient_save_path
        self.mlp_gradient_save_path = mlp_gradient_save_path
        self.mlp_activation_save_path = mlp_activation_save_path

    def training_start(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"\n🚀 [Local Rank {local_rank}] Starting single-batch gradient computation...")

        def add_rank_suffix(path):
            base, ext = os.path.splitext(path)
            return f"{base}_rank{local_rank}{ext}"

        self.mlp_activation_save_path = add_rank_suffix(self.mlp_activation_save_path)
        self.qkv_gradient_save_path = add_rank_suffix(self.qkv_gradient_save_path)
        self.mlp_gradient_save_path = add_rank_suffix(self.mlp_gradient_save_path)

        train_dataloader = self.get_train_dataloader()
        # Get the first batch from the dataloader
        batch = next(iter(train_dataloader))
        # Set the model to training mode
        self._train_batch_size = len(batch[list(batch.keys())[0]])
        model = self._wrap_model(self.model_wrapped)
        print(f"[Rank {local_rank}] Model type: {type(model)}")

        model.train()
        # Clear previous gradients
        model.zero_grad()
        # Move the batch to GPU
        inputs = self._prepare_inputs(batch)

        # -------------------------------
        # 🌟 New: Hooks for collecting MLP activations
        # -------------------------------
        mlp_activations = {}
        hooks = []
        original_model = model.module if hasattr(model, "module") else model
        # Iterate through all submodules of the model, find layers containing "mlp", and register forward hooks for their sublayers
        for name, module in original_model.named_modules():
            # 🔍 Match layers with "up_proj" in their names
            if ("gate_proj" in name) and isinstance(module, torch.nn.Linear):
                def make_hook(n):
                    def hook(module, input, output):
                        activation = output.detach().cpu()
                        mlp_activations[n] = {
                            "activation": activation,
                            "shape": activation.shape,
                            "abs_mean": activation.abs().mean().item(),
                            "layer_type": type(module).__name__
                        }
                    return hook

                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)
        print(f"📌 Registered {len(hooks)} forward hooks")

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss.mean()
        # Backward pass
        self.accelerator.backward(loss)
        torch.cuda.synchronize()
        if self.args.local_rank != -1:
            dist.barrier()
        print(f"✅ Backward pass completed. Loss: {loss.item():.4f}")

        # Collect MLP activations
        print(" --->>> Analysis and save MLP activations <<<---")
        mlp_activation_results = []
        for name, act_info in mlp_activations.items():
            act_np = act_info["activation"].detach().cpu().numpy()  # Convert to NumPy
            mlp_activation_results.append({
                "param_name": name,
                "activation": act_np,  # Original activations
                "loss": loss.item()
            })

        joblib.dump(mlp_activation_results, self.mlp_activation_save_path)
        print(f"[Local Rank {local_rank}] Saved activations to: {self.mlp_activation_save_path}")
        for h in hooks:
            h.remove()

        # Collect gradients of self_attn
        print(" --->>> Calculating self_attn and MLP layers parameter gradients <<<---")
        qkv_grads = {}
        mlp_grads = {}

        # original_model = model.module if hasattr(model, "module") else model
        for name, param in original_model.named_parameters():
            if not param.requires_grad:
                continue
            if "self_attn" in name and param.grad is not None:
                qkv_grads[name] = param.grad.detach().cpu()
            if "mlp" in name and param.grad is not None:
                mlp_grads[name] = param.grad.detach().cpu()

        # Convert to NumPy and package results
        print(" --->>> Analysis and save QKV parameter gradients <<<---")
        qkv_results = []
        for name, grad in qkv_grads.items():
            grad_np = grad.detach().cpu().numpy()
            qkv_results.append({
                "param_name": name,
                "gradient": grad_np,  # Original gradients (non-averaged)
                "loss": loss.item()
            })

        # Save results
        save_dir = os.path.dirname(self.qkv_gradient_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        joblib.dump(qkv_results, self.qkv_gradient_save_path)
        print(f"\n💾 Gradient from single batch saved to: {self.qkv_gradient_save_path}")
        print(f"   Number of self_attn params captured: {len(qkv_results)}")

        # *********************
        print(" --->>> Analysis and save MLP layers parameter gradients <<<---")
        qkv_results = []  # Clear CPU memory
        mlp_results = []
        for name, grad in mlp_grads.items():
            grad_np = grad.detach().cpu().numpy()
            mlp_results.append({
                "param_name": name,
                "gradient": grad_np,  # Original gradients (non-averaged)
                "loss": loss.item()
            })

        # Save results
        save_dir = os.path.dirname(self.mlp_gradient_save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        joblib.dump(mlp_results, self.mlp_gradient_save_path)
        print(f"\n💾 Gradient from single batch saved to: {self.mlp_gradient_save_path}")
        print(f"   Number of self_attn params captured: {len(qkv_results)}")
        print(f"   Final loss: {loss.item():.4f}")
