# Decomposing the Basic Abilities of LLMs in Multi-Task Instruct-Tuning

<p align="center">
<img src="logo.jpg" alt="logo" width="500">
</p>

This is the repository for our paper _Decomposing the Basic Abilities of LLM in Multi-Task Instruct-Tuning_

## Overview

We develop a new module in `peft` repository to implement our method and its baseline method LoRAMoE.
To achieve this, we add a new dir `loramoe` under `peft/tuners/`.
You can modify the class `peft.tuners.loramoe.LoraMoELinear` and `peft.tuners.loramoe.LoraMoEModel`.
in `peft/tuners/loramoe/` to implement your own methods based on LoRAMoE.

## Environment Setup

Following [LoRAMoE](https://github.com/Ablustrund/LoRAMoE), you can install the environment by
```shell
conda env create -f environment.yml
```
or
```shell
conda create -n loramoe python=3.10 -y
pip install -r requirements.txt
```

## Data Preparation

Put [SuperNI](https://github.com/circle-hit/SAPT/tree/master/CL_Benchmark/SuperNI) data under `data/SuperNI`.
The structure should be like:
```data/SuperNI
├── task1
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── task2
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
└── ...
```

## Training
You can run the training script as follows:
```shell
bash src/run_loramoe_group_qwen3-4b.sh
```
Make sure to modify the script according to your needs (e.g., model name, data path, hyperparameters, etc.).

To be convenient, we also provide some example scripts to run baselines and our methods in `src/` folder.

## Evaluation
You can run the evaluation script as follows:
```shell
python src/parse_scores.py --run_name output/[YOUR_RUN_NAME]  
# for the continual setting
python src/parse_scores_mtl.py --run_name output/[YOUR_RUN_NAME]  
# for the multi-task setting
```
Make sure to replace `[YOUR_RUN_NAME]` with the actual name of your run.
Also, we also provide evaluation scripts in `src/parse_scores.sh` and `src/parse_scores_mtl.sh`.

To further analyze LoRA angles and loss curves, you can use:
```shell
python src/analysis_angle.py --name output/[YOUR_RUN_NAME]
python src/analysis_loss.py --name output/[YOUR_RUN_NAME]
```

## Analysis
We provide some scripts to analysis the overlap neurons and parameters in `src/analysis_overlap.py` and `analysis/`.
1. To analyze the activated neurons and their gradients, you can run:
```shell
bash analysis/1_run_analysis.sh
```
2. Merge the results:
```shell
python analysis/2_merge_data.py --analysis_name analysis/[YOUR_RUN_NAME]
```
3. Plot and output the results:
```shell
python analysis/3_quan_analysis.py --analysis_name analysis/[YOUR_RUN_NAME]
```

## Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex

```
