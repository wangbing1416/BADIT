import sys
import os
import json
import pandas as pd
import math

# load json
def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def export_colored_latex_table(a, b, output_file='output.tex', color_name='tomato'):
    """
    Convert a 2D numeric list `a` into LaTeX table rows. Each number x is formatted with two decimals
    and wrapped as \cellcolor{color_name!xx.xx}xx.xx. Each row is prefixed by the corresponding
    element in `b`, separated with & and written to a .tex file.

    Args:
        a (list[list[float|int]]): 2D list of numbers.
        b (list): Prefix list with the same number of rows as `a`.
        output_file (str): Output file path, default 'output.tex'.
        color_name (str): LaTeX color name, e.g., 'tomato'.

    Returns:
        None
    """
    if len(a) != len(b):
        raise ValueError("Lists a and b must have the same number of rows.")

    lines = []
    for row_a, prefix in zip(a, b):
        # Format each number to one decimal place string (e.g., 41 → "41.0")
        formatted_vals = [f"{val:.1f}" for val in row_a]
        # Build colored LaTeX cells
        colored_cells = [f"\\cellcolor{{{color_name}!{val}}}{val}" for val in formatted_vals]
        # Concatenate prefix and data row
        full_row = [str(prefix)] + colored_cells
        line = " & ".join(full_row) + " \\\\"
        lines.append(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table saved to {output_file}")


def cal_continue_learning_metrics(scores_array, baseline_array):
    task_num = len(scores_array)

    Cl = sum(scores_array[-1]) / task_num

    fgt_list = []
    for t_idx in range(task_num - 1):
        history = [line[t_idx] for line in scores_array[:-1]]
        history_best = max(history)
        fgt_list.append(history_best - scores_array[-1][t_idx])
    Fgt = sum(fgt_list) / len(fgt_list)

    if len(baseline_array) > 0:
        Fwt = sum([scores_array[i][i] for i in range(task_num)]) / task_num - sum(baseline_array) / task_num
    else:
        Fwt = 0

    Bwt = sum([scores_array[-1][i] - scores_array[i][i] for i in range(task_num)]) / task_num

    return {
        'Cl': Cl,
        'Fgt': Fgt,
        'Fwt': Fwt,
        'Bwt': Bwt,
    }


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='output/gemma2-9b-loramoe-seed3')
    args = parser.parse_args()

    task_order = load_json(f'./{args.run_name}/task_order.json')[0]
    task_num = len(task_order)
    results = load_json(f'./{args.run_name}/eval_results.jsonl')
    results = sorted(results, key=lambda x: task_order.index(x['training_task_name']))

    baseline_root = f"{args.run_name.split('-seed')[0].split('-r4')[0]}-baseline"
    try:
        baselines = [load_json(f"./{baseline_root}/eval_results_{task}.jsonl") for task in task_order]
    except:
        baselines = [None for task in task_order]

    prefix = 'eval_rougeL_for_'
    # filter the useful values
    filtered_results = []
    baseline_results = []
    for t, res, basel in zip(task_order, results, baselines):
        fil_result = {}
        for task_i, task in enumerate(task_order):
            fil_result[prefix + task] = res[prefix + task]
        filtered_results.append(fil_result)
        if basel:
            baseline_results.append(basel[0][prefix + t])

    scores = []
    for task_i, task in enumerate(task_order):
        scores.append(list(filtered_results[task_i].values()))

    export_colored_latex_table(scores, task_order, f'./{args.run_name}/continual_learning_scores.tex')

    metrics = cal_continue_learning_metrics(scores, baseline_results)

    output_path = f"./{args.run_name}/final_eval_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        for res in filtered_results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"**********\n"
          f"Final eval metrics: {metrics} \n"
          f"the file has been saved in {output_path} \n"
          f"**********\n")
