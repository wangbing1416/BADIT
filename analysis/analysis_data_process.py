import json
import os
import random
from datasets import Dataset, concatenate_datasets
from dataclasses import dataclass
import transformers
from transformers import PreTrainedTokenizer
from typing import Union, Optional, List, Dict, Sequence
import torch
import logging

logger = logging.getLogger(__name__)

# Global constants (can be provided externally)
PROMPT_TEMPLATE = "{instruction}"
IGNORE_INDEX = -100


def check_path(path):
    if isinstance(path, list):
        for p in path:
            if not p or not os.path.exists(p):
                raise ValueError(f'{p} is not valid, please check the input path!')

    if not path or not os.path.exists(path):
        raise ValueError(f'{path} is not valid, please check the input path!')


def build_single_instruction_dataset_from_json(
    data_file,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    model_name: str,
    max_num_instances: Optional[int] = None,
    data_cache_dir: Optional[str] = None,
    preprocessing_num_workers: Optional[int] = None,
    task_type: int = None,  # Optional custom task type
) -> Dataset:
    """
    Build an instruction dataset from a single JSON file. The output format matches
    `build_instruction_dataset`.

    Args:
        data_file: Path to the JSON file, containing `Definition` and `Instances`.
        tokenizer: Hugging Face tokenizer.
        max_seq_length: Maximum sequence length.
        max_num_instances: Maximum number of instances to keep.
        data_cache_dir: Directory for caching processed dataset.
        preprocessing_num_workers: Number of workers for preprocessing (multiprocessing).
        task_type: Optional task type label (e.g., "CL").

    Returns:
        Dataset: A Hugging Face Dataset in torch format containing input_ids, labels, and task_types.
    """
    logging.info(f"Building instruction dataset from {data_file}...")

    # Check the file path
    check_path(data_file)

    # Read JSON data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    subset = 'test' if 'test' in data_file or 'dev' in data_file else 'train'

    definition = ""
    if data.get("Definition"):
        if isinstance(data["Definition"], list):
            definition = data["Definition"][0].strip()
        else:
            definition = data["Definition"].strip()
        definition += "\n\n"

    # Build list of examples
    examples = []
    count = 0
    for idx, instance in enumerate(data.get("Instances", [])):
        instruction = definition + "Now complete the following example -\n"
        instruction += f"Input: {instance['input']}\nOutput: "
        # messages = [
        #     {"role": "user", "content": f"Input: {definition + "Now complete the following example -\n" + instance['input']}\nOutput: "},
        # ]
        # instruction = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)
        # instruction = instruction['input_ids'][0]

        if isinstance(instance["output"], list) and len(instance["output"]) > 0:
            output = instance["output"][random.randint(0, len(instance["output"]) - 1)]
        else:
            output = instance["output"]

        examples.append({
            "instruction": instruction,
            "input": "",  # Already included in the instruction
            "output": output,
            "task_type": task_type,
            "subsets": subset,
        })
        count += 1
        if max_num_instances and count >= max_num_instances:
            break
    # print(f'---->>>>tokenizer.decode(instruction): {tokenizer.decode(instruction)}')

    if len(examples) == 0:
        raise ValueError(f"No instances found in {data_file}")

    # Convert to a Hugging Face Dataset
    raw_dataset = Dataset.from_list(examples)

    # Tokenization function (reproduce original logic)
    def tokenization(examples):
        sources = []
        targets = []
        task_types = []
        subsets = []
        prompt = PROMPT_TEMPLATE

        for instruction, input_text, output, task_type, subset in zip(
            examples['instruction'], examples['input'], examples['output'], examples['task_type'], examples['subsets']
        ):
            # Merge instruction and input (if any)
            full_instruction = instruction
            if input_text and input_text.strip():
                full_instruction = f"{full_instruction}\n{input_text}"
            source = prompt.format_map({'instruction': full_instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            # sources.append(instruction)

            targets.append(target)
            task_types.append(task_type)
            subsets.append(subset)

        # Tokenize source (do not return attention mask)
        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        # tokenized_sources = sources
        # Tokenize target (do not add special tokens to avoid duplicate eos)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        # logger.info(f'---->>>>tokenizer.decode(tokenized_sources[0]): {tokenizer.decode(tokenized_sources[0])}')

        all_input_ids = []
        all_input_ids_wo_labels = []
        all_labels = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = s + t
            input_ids_wo_labels = s
            labels = [IGNORE_INDEX] * len(s) + t  # Compute loss only on the output portion

            # Truncation
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
                input_ids_wo_labels = input_ids_wo_labels[:max_seq_length]
                labels = labels[:max_seq_length]

            all_input_ids.append(torch.LongTensor(input_ids))
            all_input_ids_wo_labels.append(torch.LongTensor(input_ids_wo_labels))
            all_labels.append(torch.LongTensor(labels))

        return {
            'sources': sources,
            'targets': targets,
            'input_ids': all_input_ids,
            'labels': all_labels,
            'input_ids_wo_labels': all_input_ids_wo_labels,
            'task_types': task_types,
            'subsets': subsets,
        }

    # Cache path: based on filename + max_seq_length
    # clear cache: find ./SuperNI -type d -name "cache" -exec rm -rf {} +
    if data_cache_dir is None:
        data_cache_dir = os.path.join(os.path.dirname(data_file), f"cache_{model_name}")
    os.makedirs(data_cache_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(data_file))[0]
    cache_path = os.path.join(data_cache_dir, f"{file_name}_{max_seq_length}")

    try:
        processed_dataset = Dataset.load_from_disk(cache_path)
        logger.info(f"Dataset loaded from cache: {cache_path}")
    except Exception as e:
        logger.info(f"Cache not found or invalid: {cache_path}. Processing...")
        processed_dataset = raw_dataset.map(
            tokenization,
            batched=True,
            num_proc=preprocessing_num_workers,
            remove_columns=["instruction", "input", "output", "task_type", "subsets"],
        )
        processed_dataset.save_to_disk(cache_path)

    # Set dataset format to torch
    processed_dataset.set_format('torch')

    return processed_dataset


def build_instruction_dataset_from_json(
    data_file,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    model_name: str,
    max_num_instances: Optional[int] = None,
    data_cache_dir: Optional[str] = None,
    preprocessing_num_workers: Optional[int] = None,
    task_type=None,  # Optional custom task type
) -> Dataset:

    if not isinstance(data_file, (list, tuple)):
        data_file = [data_file]
    if not isinstance(task_type, (list, tuple)):
        task_type = [task_type]
    assert len(data_file) == len(task_type)

    all_datasets = []
    for file, task_t in zip(data_file, task_type):
        all_datasets.append(build_single_instruction_dataset_from_json(file, tokenizer, max_seq_length, model_name,
                                                                       max_num_instances, data_cache_dir,
                                                                       preprocessing_num_workers, task_t))
    return concatenate_datasets(all_datasets)


def left_pad_sequence(sequences, batch_first=False, padding_value=0):
    """
    Left-pad sequences: padding is added to the front of each sequence.
    """
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)

    padded_sequences = []
    for seq in sequences:
        # Pad on the left
        pad_length = max_len - len(seq)
        padded_seq = torch.cat(
            [torch.full((pad_length,), padding_value, dtype=seq.dtype, device=seq.device), seq]
        )
        padded_sequences.append(padded_seq)

    if batch_first:
        return torch.stack(padded_sequences)
    else:
        return torch.nn.utils.rnn.pad_sequence(padded_sequences, batch_first=False, padding_value=padding_value)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources, targets, input_ids, labels, input_ids_wo_labels, task_types, subsets = tuple(
            [instance[key] for instance in instances] for key in
            ("sources", "targets", "input_ids", "labels", "input_ids_wo_labels", "task_types", "subsets"))
        input_ids = left_pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids_wo_labels = left_pad_sequence(
            input_ids_wo_labels, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = left_pad_sequence(labels, batch_first=True, padding_value=-100)
        # labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        # task_types = torch.tensor(task_types)

        return dict(
            sources=sources,
            targets=targets,
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            input_ids_wo_labels=input_ids_wo_labels,
            attention_mask_wo_labels=input_ids_wo_labels.ne(self.tokenizer.pad_token_id),
            task_types=task_types
        )