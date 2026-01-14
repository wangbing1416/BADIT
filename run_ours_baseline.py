#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import random
import bitsandbytes
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import datasets
import torch
import torch.distributed as dist
import json
import shutil
import numpy as np
import transformers
from peft import LoraMoEConfig
from peft.tuners.loramoe import LoraMoELinear
from transformers import (
    CONFIG_MAPPING,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from peft import LoraConfig, TaskType, get_peft_model, PeftModel, get_peft_model_state_dict

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from data_process import build_instruction_dataset_from_json, DataCollatorForSupervisedDataset
from build_trainer import CLTrainer, skip_instructions
from compute_metrics import compute_metrics, compute_grouped_metrics

CUDA_VISIBLE_DEVICES = 0
CUDA_LAUNCH_BLOCKING = 1
LOW_RANK = 0
torch.cuda.set_device(LOW_RANK)
os.environ["WANDB_MODE"] = "offline"
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        self.save_model(args, state, kwargs)
        # get the default saved path：output_dir/checkpoint-xxx
        default_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        if not os.path.exists(default_checkpoint_dir):
            return control

        # define the checkpoint name
        custom_checkpoint_name = f"checkpoint-{args.task_name}-step{state.global_step}"
        custom_checkpoint_dir = os.path.join(args.output_dir, custom_checkpoint_name)

        if os.path.exists(custom_checkpoint_dir):
            shutil.rmtree(custom_checkpoint_dir)
        shutil.move(default_checkpoint_dir, custom_checkpoint_dir)

        print(f"Checkpoint saved as: {custom_checkpoint_name}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)


class RegroupLoraMoECallback(transformers.TrainerCallback):
    def __init__(self, regroup_logger, metrics_logger):
        self.regroup_logger = regroup_logger
        self.metrics_logger = metrics_logger

    def on_epoch_end(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        current_epoch = int(state.epoch)
        max_epochs = args.num_train_epochs

        # === Step 1: Save metrics to metrics.log ===
        # logs may be None, but the latest entry in state.log_history usually holds data
        last_log = {}
        if hasattr(state, 'log_history') and state.log_history:
            last_log = state.log_history[-1]  # dict like {'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}

        # Construct a readable log string
        log_str = ", ".join(f"{k}: {v:.6g}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in last_log.items())
        if log_str:
            self.metrics_logger.info(f"[Epoch {current_epoch}] {log_str}")

        # === Step 2: Run regroup ===
        for name, module in model.base_model.model.named_modules():
            if isinstance(module, LoraMoELinear):
                grad_cache = module.get_last_computed_gradients()
                if not grad_cache:
                    continue
                try:
                    self.regroup_logger.info(f">> EPOCH: {current_epoch}")
                    module.regroup_by_cached_gradients(
                        grad_cache,
                        "default",
                        self.regroup_logger,
                        name,
                        current_epoch=current_epoch,
                        max_epochs=max_epochs
                    )
                    if hasattr(module, '_last_grads_cache'):
                        module._last_grads_cache.clear()
                except Exception as e:
                    logger.error(f"Regroup failed: {e}")


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='../huggingface/llama3.2-3b-instruct',
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        default='../huggingface/llama3.2-3b-instruct',
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation: 'eager', 'sdpa', 'flash_attention_2'"},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})

    max_seq_length: Optional[int] = field(default=1024)
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    task_name: Optional[str] = field(default=None, metadata={"help": "The datasets processed stored"})


@dataclass
class MyTrainingArguments(Seq2SeqTrainingArguments):
    predict_with_generate: Optional[bool] = field(
        default=True,
    )
    lora_config_file: Optional[str] = field(default='config/loramoe.config')
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    flash_attn: Optional[bool] = field(default=False)
    double_quant: Optional[bool] = field(default=True)
    quant_type: Optional[str] = field(default="nf4")
    load_in_kbits: Optional[int] = field(default=16)

    learning_rate: float = field(default=0.0002, metadata={"help": "The initial learning rate for AdamW."})


def find_all_linear_names(args, model):
    cls = bitsandbytes.nn.Linear4bit if args.load_in_kbits == 4 else (
        bitsandbytes.nn.Linear8bitLt if args.load_in_kbits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    file_handler = logging.FileHandler(f'{training_args.output_dir}/logging.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                       datefmt="%m/%d/%Y %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                         datefmt="%m/%d/%Y %H:%M:%S")  # Or keep consistent format
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Dedicated logger: write to file only, not output to console
    regroup_logger = logging.getLogger("regroup_logger")
    regroup_logger.setLevel(logging.INFO)
    # Avoid adding handler repeatedly
    if not regroup_logger.handlers:
        regroup_file_handler = logging.FileHandler(f'{training_args.output_dir}/regroup_logger.log')
        regroup_file_handler.setLevel(logging.INFO)
        regroup_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                                              datefmt="%m/%d/%Y %H:%M:%S")
        regroup_file_handler.setFormatter(regroup_formatter)
        regroup_logger.addHandler(regroup_file_handler)
        # Key point: do not add StreamHandler！

    # In main(), after the logging configuration section:
    metrics_logger = logging.getLogger("loss_logger")
    metrics_logger.setLevel(logging.INFO)
    # Avoid adding handler repeatedly
    if not metrics_logger.handlers:
        metrics_file_handler = logging.FileHandler(f'{training_args.output_dir}/loss_logger.log')
        metrics_file_handler.setLevel(logging.INFO)
        metrics_formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
        metrics_file_handler.setFormatter(metrics_formatter)
        metrics_logger.addHandler(metrics_file_handler)
        # Do not add StreamHandler → Do not output to console

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        logger.info('last_checkpoint', last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if 'llama' in model_args.model_name_or_path.lower():
        config.bos_token_id, tokenizer.bos_token_id = 1, 1
        config.eos_token_id, tokenizer.eos_token_id = 2, 2
        config.pad_token_id, tokenizer.pad_token_id = 1, 1
        tokenizer.padding_side = "left"  # Left padding is safer for generation tasks
    logger.info(
        f"----->>>>tokenizer.bos_token / bos_token_id: {tokenizer.bos_token} / {tokenizer.bos_token_id}")  # <|begin_of_text|>
    logger.info(
        f"----->>>>tokenizer.eos_token / eos_token_id: {tokenizer.eos_token} / {tokenizer.eos_token_id}")  # <|eot_id|>
    logger.info(f"----->>>>tokenizer.pad_token / pad_token_id: {tokenizer.pad_token} / {tokenizer.pad_token_id}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            train_task_list = []
            for name in os.listdir(path):
                if os.path.isdir(os.path.join(path, name)):
                    train_task_list.append(name)
            random.seed(training_args.seed)
            random.shuffle(train_task_list)
            with open(os.path.join(training_args.output_dir, f"task_order.json"), "w") as fout:
                fout.write(json.dumps(train_task_list))
            logger.info(f"Training task names: {' \n'.join(train_task_list)}")
            task = data_args.task_name
            file = os.path.join(path, f"{task}/train.json")
            train_dataset = build_instruction_dataset_from_json(
                data_file=file,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                model_name=model_args.model_name_or_path.split("/")[-1],
                data_cache_dir=None,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                task_type=task, )
            logger.info(f"TASK {task} have {len(train_dataset)} training samples")
    if training_args.do_eval:
        # todo: add a dataset testing general abilities!
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            task_list = []
            for name in os.listdir(path):
                if os.path.isdir(os.path.join(path, name)):
                    task_list.append(name)
            logger.info(f"Testing task names: {' '.join(task_list)}")
            task = data_args.task_name
            file = os.path.join(path, f"{task}/test.json")
            eval_dataset = build_instruction_dataset_from_json(
                data_file=file,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                model_name=model_args.model_name_or_path.split("/")[-1],
                data_cache_dir=None,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
                task_type=task, )
            logger.info(f"eval TASKs have {len(eval_dataset)} testing samples")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(',')
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )
    model.enable_input_require_grads()
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    model.config.use_cache = False

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer) or model.config.vocab_size != len(tokenizer):
        logger.info(f"Resize model vocab size to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    if training_args.peft_path is not None:  # --------------------------> train from the trained lora model
        logger.info("Peft from pre-trained model")

        model = PeftModel.from_pretrained(model, training_args.peft_path)
    else:  # train from the sketch
        logger.info("Init new peft model")

        lora_config_file = training_args.lora_config_file
        with open(lora_config_file, "r", encoding="utf-8") as f:
            lora_config = json.load(f)

        # ---------- START: modify peft linear in model ----------
        peft_config = LoraMoEConfig(**lora_config)
        logger.info(f"--->>> peft_config:{peft_config.to_dict()}")

        model = get_peft_model(model, peft_config)
        # ---------- END: modify peft linear in model ----------

    # todo: add max new token parameters
    model.generation_config.max_new_tokens = data_args.max_target_length
    model.generation_config.max_length = data_args.max_target_length

    if training_args.gradient_checkpointing and \
            (not model.modules_to_save or 'embed_tokens' not in model.modules_to_save):
        # enable requires_grad to avoid exception during backward pass when using gradient_checkpoint without tuning embed.
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):
            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)

            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    for name, module in model.named_modules():
        if isinstance(module, LoraMoELinear):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
        if 'norm' in name:
            module = module.to(torch.float16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                if training_args.fp16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.float16)

    model.print_trainable_parameters()
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    for name, parameters in model.named_parameters():
        logger.info(f"{name}, :, {parameters.size()},{parameters.requires_grad}")

    training_args.remove_unused_columns = False

    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = skip_instructions(model, [e["labels"] for e in dataset], tokenizer)
        logger.info(f"------>>>>>decoded_preds[0]: {decoded_preds[0]}")
        logger.info(f"------>>>>>references[0]: {references[0]}")
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["task_types"])
        result.update(result_per_task)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "task_types": example["task_types"],
                        "input_ids": example["sources"],
                        "labels": example["targets"],
                        "Prediction": pred
                    }) + "\n")
        return result

    def is_rank_0():
        return not dist.is_initialized() or dist.get_rank() == 0

    # Initialize our Trainer
    trainer = CLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
    )
    trainer.add_callback(SavePeftModelCallback)
    trainer.add_callback(RegroupLoraMoECallback(regroup_logger, metrics_logger))
    trainer.args.task_name = data_args.task_name

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset, **model.generation_config.to_dict())
        metrics["training_task_name"] = data_args.task_name
        if is_rank_0():
            eval_results_path = os.path.join(training_args.output_dir, f"eval_results_{data_args.task_name}.jsonl")
            with open(eval_results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            logger.info(f"the evaluation results are appended to {eval_results_path}")


if __name__ == "__main__":
    main()
