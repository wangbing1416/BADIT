import os
import argparse

import joblib
import torch
from functools import partial
from pathlib import Path
import numpy as np
import pickle
import zipfile
from collections import defaultdict
from tqdm import tqdm

tasks = [
    "task1590_diplomacy_text_generation",
    "task875_emotion_classification",
    "task511_reddit_tifu_long_text_summarization",
    "task1572_samsum_summary",
    "task591_sciq_answer_generation",
    "task002_quoref_answer_generation",
    "task639_multi_woz_user_utterance_generation",
    "task748_glucose_reverse_cause_event_detection",  # load error
    "task1290_xsum_summarization",
    "task1510_evalution_relation_extraction",  # load error
    "task363_sst2_polarity_classification",
    "task181_outcome_extraction",
    "task1687_sentiment140_classification",  # load error
    "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation"  # load error
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--root_path', type=str, default='analysis/')
    parser.add_argument('--merge_mlp_activation', type=bool, default=False)
    parser.add_argument('--merge_mlp_grad', type=bool, default=False)
    parser.add_argument('--merge_qkv_grad', type=bool, default=True)
    parser.add_argument('--analysis_activation', type=bool, default=False)
    parser.add_argument('--analysis_name', type=str, default='qwen3-4b-analysis-1026')
    args = parser.parse_args()

    analysis_path = Path(os.path.join(args.root_path, args.analysis_name))
    task_to_path = {}
    for subdir in analysis_path.iterdir():
        if not subdir.is_dir():
            continue
        dir_name = subdir.name  # e.g., avg_grads_task875_emotion_classification_llama3.2-3b-instruct

        # 尝试匹配 tasks 中的每一个任务名
        matched = False
        for task in tasks:
            if task in dir_name:
                task_to_path[task] = str(subdir.name)  # save absolute path
                break  # one dir for only one task

    if args.merge_mlp_activation:
        print(f"***** Merging MLP ACTIVATION (gate)! *****")
        for task in tqdm(task_to_path):
            mlp_activation_save_paths = [os.path.join(analysis_path, task_to_path[task], f'mlp_param_activation_rank{r}.pt')
                                         for r in range(args.rank)]
            processed_mlp_activation_save_paths = os.path.join(analysis_path, task_to_path[task], f'mlp_gate_activation.pkl')

            activations_by_name = defaultdict(list)
            for mlp_activation_save_path in mlp_activation_save_paths:
                mlp_gate_activation = joblib.load(mlp_activation_save_path)
                for item in mlp_gate_activation:
                    name = item['param_name']
                    activation = item['activation']  # shape: (32, 109, 8192)
                    activations_by_name[name].append(activation)

            result_vectors = defaultdict(list)
            for name, activations in activations_by_name.items():
                processed = []
                for act in activations:
                    # act.shape = (32, L_i, 8192)
                    # average over sequence length dimension (axis=1) → get (32, 8192)
                    mean_over_tokens = np.mean(act, axis=1)  # squeeze the L_i dimension
                    processed.append(mean_over_tokens)  # list of (32, 8192)
                # stack all samples (32, 8192) to (N, 32, 8192)
                stacked = np.stack(processed, axis=0)  # shape: (N, 32, 8192)
                # average over all samples and batch dimensions
                final_vector = np.mean(stacked, axis=(0, 1))  # average over axis 0 and 1 → shape: (8192,)
                result_vectors[name] = final_vector

            joblib.dump(result_vectors, processed_mlp_activation_save_paths)
            print(f"The processed activation file is saved in {processed_mlp_activation_save_paths}")

    if args.merge_mlp_grad:
        print(f"***** Merging MLP GRADIENT (gate)! *****")
        for task in tqdm(task_to_path):
            mlp_gradient_save_paths = [os.path.join(analysis_path, task_to_path[task], f'mlp_param_grads_avg_rank{r}.pt')
                                       for r in range(args.rank)]
            processed_mlp_grad_save_paths = os.path.join(analysis_path, task_to_path[task], f'mlp_gate_grad.pkl')

            grads_by_name = defaultdict(list)
            for mlp_activation_save_path in mlp_gradient_save_paths:  # process per rank
                mlp_gate_activation = joblib.load(mlp_activation_save_path)
                for item in mlp_gate_activation:
                    name = item['param_name']
                    if 'gate_proj' not in name:
                        continue
                    grad = item['gradient']  # shape: (4096, 8192)
                    grads_by_name[name].append(grad)

            result_vectors = defaultdict(list)
            for name, grads in grads_by_name.items():
                # stack all samples to (N, 4096, 8192), then average over axis 0
                averaged_grad = np.mean(np.stack(grads, axis=0), axis=0)  # shape: (4096, 8192)
                result_vectors[name] = averaged_grad

            joblib.dump(result_vectors, processed_mlp_grad_save_paths)
            print(f"The processed activation file is saved in {processed_mlp_grad_save_paths}")

    if args.merge_qkv_grad:
        print(f"***** Merging QKV GRADIENT! *****")
        for task in tqdm(task_to_path):
            qkv_gradient_save_paths = [
                os.path.join(analysis_path, task_to_path[task], f'qkv_param_grads_avg_rank{r}.pt')
                for r in range(args.rank)]
            processed_q_grad_save_paths = os.path.join(analysis_path, task_to_path[task], f'q_gate_grad.pkl')
            processed_k_grad_save_paths = os.path.join(analysis_path, task_to_path[task], f'k_gate_grad.pkl')
            processed_v_grad_save_paths = os.path.join(analysis_path, task_to_path[task], f'v_gate_grad.pkl')

            q_grads_by_name = defaultdict(list)
            k_grads_by_name = defaultdict(list)
            v_grads_by_name = defaultdict(list)
            for qkv_activation_save_path in qkv_gradient_save_paths:  # process per rank
                qkv_gate_activation = joblib.load(qkv_activation_save_path)
                for item in qkv_gate_activation:
                    name = item['param_name']
                    if 'q_proj' in name:
                        grad = item['gradient']  # shape: (3072, 3072)
                        q_grads_by_name[name].append(grad)
                    if 'k_proj' in name:
                        grad = item['gradient']  # shape: (1024, 3072)
                        k_grads_by_name[name].append(grad)
                    if 'v_proj' in name:
                        grad = item['gradient']  # shape: (1024, 3072)
                        v_grads_by_name[name].append(grad)

            result_vectors = defaultdict(list)
            for name, grads in q_grads_by_name.items():
                averaged_grad = np.mean(np.stack(grads, axis=0), axis=0)  # shape: (4096, 8192)
                result_vectors[name] = averaged_grad
            joblib.dump(result_vectors, processed_q_grad_save_paths)
            print(f"The processed Q grad file is saved in {processed_q_grad_save_paths}")

            result_vectors = defaultdict(list)
            for name, grads in k_grads_by_name.items():
                averaged_grad = np.mean(np.stack(grads, axis=0), axis=0)  # shape: (4096, 8192)
                result_vectors[name] = averaged_grad
            joblib.dump(result_vectors, processed_k_grad_save_paths)
            print(f"The processed K grad file is saved in {processed_k_grad_save_paths}")

            result_vectors = defaultdict(list)
            for name, grads in v_grads_by_name.items():
                averaged_grad = np.mean(np.stack(grads, axis=0), axis=0)  # shape: (4096, 8192)
                result_vectors[name] = averaged_grad
            joblib.dump(result_vectors, processed_v_grad_save_paths)
            print(f"The processed V grad file is saved in {processed_v_grad_save_paths}")
