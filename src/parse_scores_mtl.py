import sys
import os
import json

# load json
def load_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def cal_mtl_metrics(scores_array, baseline_array):
    task_num = len(scores_array)

    rouge = sum(scores_array) / task_num
    colla = sum(scores_array) / task_num - sum(baseline_array) / task_num

    return {
        'rouge': rouge,
        'colla': colla,
    }


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='output/qwen3-8b-olora-mtl-seed3')
    args = parser.parse_args()

    task_order = load_json(f'./{args.run_name}/task_order.json')[0]
    task_num = len(task_order)
    results = load_json(f'./{args.run_name}/eval_results.jsonl')[0]

    baseline_root = f"{args.run_name.split('-seed')[0].split('-mtl')[0].replace('loramoe-group', 'loramoe_group')}-baseline"
    try:
        baselines = [load_json(f"./{baseline_root}/eval_results_{task}.jsonl") for task in task_order]
    except:
        baselines = [None for task in task_order]

    prefix = 'eval_rougeL_for_'
    # filter the useful values
    filtered_results = []
    baseline_results = []
    for t, basel in zip(task_order, baselines):
        filtered_results.append(results[prefix + t])
        if basel:
            baseline_results.append(basel[0][prefix + t])

    metrics = cal_mtl_metrics(filtered_results, baseline_results)

    output_path = f"./{args.run_name}/final_eval_results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        f.write(json.dumps({'mtl_results': filtered_results,
                            'mtl_baseline': baseline_results,
                            'mtl_task_order': task_order}, ensure_ascii=False) + "\n")
    print(f"**********\n"
          f"Final eval metrics: {metrics} \n"
          f"the file has been saved in {output_path} \n"
          f"**********\n")
