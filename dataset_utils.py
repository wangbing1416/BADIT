import json
import os
import random
from datasets import DatasetDict, Dataset
from hashlib import md5

# === Copy the core logic of cl_dataset.py ===

TASK_CONFIG_FILES = {"train": "train_tasks.json", "dev": "dev_tasks.json", "test": "test_tasks.json"}

def gen_cache_path(cache_dir, data_args):
    hash_str = data_args.data_dir + data_args.task_config_dir + \
               str(data_args.max_num_instances_per_task) + str(data_args.max_num_instances_per_eval_task)
    hash_obj = md5(hash_str.encode("utf-8"))
    hash_id = hash_obj.hexdigest()
    cache_path = os.path.join(cache_dir, str(hash_id))
    return cache_path

def check_path(path):
    if not path or not os.path.exists(path):
        raise ValueError(f'{path} is not valid, please check the input path!')

def _parse_task_config(task_config_dir):
    if not task_config_dir:
        return None
    task_configs = {}
    for task, file_name in TASK_CONFIG_FILES.items():
        task_config_file = os.path.join(task_config_dir, file_name)
        check_path(task_config_file)
        with open(task_config_file, 'r') as f:
            task_configs[task] = json.load(f)
    return task_configs

def _load_dataset(dataset_path):
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)

def load_LongSeq_dataset(dataset_path, dataset_name, max_num_instances, subset):
    data = _load_dataset(dataset_path)
    definition = ""
    if data.get("Definition"):
        if isinstance(data["Definition"], list):
            definition = data["Definition"][0].strip()
        else:
            definition = data["Definition"].strip()
        definition += "\n"

    for idx, instance in enumerate(data['Instances']):
        instruction = definition + "Now complete the following example -\n"
        instruction += "Input: {0}\nOutput: "

        if isinstance(instance["output"], list):
            label = instance["output"][random.randint(0, len(instance["output"])-1)]
        else:
            label = instance["output"]

        yield {
            "Task": "CL",
            "Dataset": dataset_name,
            "subset": subset,
            "Instance": {
                "id": str(idx),
                "sentence": instance['input'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }
        }

def load_SuperNI_dataset(dataset_path, dataset_name, max_num_instances, subset):
    data = _load_dataset(dataset_path)
    definition = ""
    if data.get("Definition"):
        if isinstance(data["Definition"], list):
            definition = "Definition: " + data["Definition"][0].strip()
        else:
            definition = "Definition: " + data["Definition"].strip()
        definition += "\n\n"

    for idx, instance in enumerate(data['Instances']):
        instruction = definition + "Now complete the following example -\n"
        instruction += "Input: {0}\nOutput: "

        if isinstance(instance["output"], list):
            label = instance["output"][random.randint(0, len(instance["output"])-1)]
        else:
            label = instance["output"]

        yield {
            "Task": "CL",
            "Dataset": dataset_name,
            "subset": subset,
            "Instance": {
                "id": str(idx),
                "sentence": instance['input'],
                "label": label,
                "ground_truth": label,
                "instruction": instruction
            }
        }

def load_cl_dataset(data_dir, task_config_dir, max_num_instances_per_task, max_num_instances_per_eval_task):
    task_configs = _parse_task_config(task_config_dir)
    check_path(data_dir)

    splits = {}

    for split_name in ["train", "dev", "test"]:
        examples = []
        task_config = task_configs[split_name]
        max_num = (max_num_instances_per_task if split_name == "train"
                   else max_num_instances_per_eval_task if split_name == "validation"
                   else None)

        for task in task_config:
            load_func = {
                'SuperNI': load_SuperNI_dataset,
                'Long_Sequence': load_LongSeq_dataset
            }.get(task)
            if not load_func:
                raise ValueError(f"Unsupported task: {task}")

            for dataset in task_config[task]:
                ds_name = dataset["dataset name"]
                # subset = "test" if split_name == "test" else split_name
                ds_path = os.path.join(data_dir, task, ds_name, f"{split_name}.json")
                check_path(ds_path)

                for sample in load_func(ds_path, ds_name, max_num, split_name):
                    examples.append(sample)

        # Shuffle and limit
        random.shuffle(examples)
        if max_num and len(examples) > max_num:
            examples = examples[:max_num]

        # Convert to Hugging Face Dataset
        datasets_list = {k: [ex[k] for ex in examples] for k in examples[0].keys()}
        splits[split_name] = Dataset.from_dict(datasets_list)

    return DatasetDict(splits)
