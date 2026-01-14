import argparse
import os
from pathlib import Path
from tqdm import tqdm
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from matplotlib.colors import ListedColormap

tasks = [
    "task1590_diplomacy_text_generation",
    "task875_emotion_classification",
    "task511_reddit_tifu_long_text_summarization",
    "task1572_samsum_summary",
    "task591_sciq_answer_generation",
    "task002_quoref_answer_generation",
    "task639_multi_woz_user_utterance_generation",
    "task748_glucose_reverse_cause_event_detection",
    "task1290_xsum_summarization",
    "task1510_evalution_relation_extraction",
    "task363_sst2_polarity_classification",
    "task181_outcome_extraction",
    "task1687_sentiment140_classification",
    "task1729_personachat_generate_next",
    "task073_commonsenseqa_answer_generation"
]


def keep_top_percent(arr, percent=10, pad=-1):
    threshold = np.percentile(arr, 100 - percent)  # 90-th percentile, i.e., top 10% threshold
    result = np.where(arr >= threshold, arr, pad)
    return result


def binarize_top_percent(arr, percent=10):
    if not (0 < percent <= 100):
        raise ValueError("percent 应该在 (0, 100] 范围内")
    threshold = np.percentile(arr, 100 - percent)  # percent=10 -> 90th percentile
    result = np.where(arr >= threshold, 1, 0).astype(int)
    return result


def plot_heatmap_with_compression(data, compression_ratio=4, output_file=None, cmap='viridis', figsize=(12, 8)):
    """
    draw heatmap with compression and column mean bar plot
    """
    # 转换为 numpy 数组
    number = len(data)
    length = len(data[0])
    data = np.array(data)

    if not (isinstance(compression_ratio, int) and compression_ratio >= 1):
        raise ValueError("compression_ratio must be a positive integer")
    if compression_ratio > length:
        raise ValueError(f"compression_ratio can not larger than {length}")

    # Truncate to length divisible by compression_ratio
    truncated_length = (length // compression_ratio) * compression_ratio
    data_truncated = data[:, :truncated_length]

    # Compress by averaging every 'compression_ratio' elements
    compressed_data = data_truncated.reshape(number, -1, compression_ratio).mean(axis=2)
    n_cols = compressed_data.shape[1]  # number of columns after compression

    # normalize or binarize
    compressed_data = binarize_top_percent(compressed_data, percent=10)
    # compressed_data = (compressed_data - compressed_data.min()) / (compressed_data.max() - compressed_data.min())

    # Calculate the mean value for each column across all tasks
    column_means = compressed_data.mean(axis=0)  # shape: (n_cols,)

    # Create two subplots vertically aligned, sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.3})

    # --- Subplot 1: Heatmap ---
    im = ax1.imshow(compressed_data, aspect='auto', cmap=cmap, origin='upper')
    ax1.set_title(f'Heatmap (Compressed by ratio {compression_ratio}, Truncated to {truncated_length})', fontsize=12)
    ax1.set_ylabel('Task Index (0 to {})'.format(number - 1))
    ax1.set_xticks([])  # Hide x ticks of heatmap to avoid clutter
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Activation Value')

    # --- Subplot 2: Bar Plot (Column Means) ---
    x_positions = np.arange(n_cols)
    ax2.bar(x_positions, column_means, width=1.0, color='steelblue', edgecolor='none')
    ax2.set_title('Average Value Across Tasks per Column', fontsize=10)
    ax2.set_xlabel('Compressed Position Index')
    ax2.set_ylabel('Mean Value')
    ax2.set_xlim(-0.5, n_cols - 0.5)  # align bars

    # optional: show fewer x-ticks if too many columns
    if n_cols <= 50:
        ax2.set_xticks(x_positions)
    else:
        step = max(1, n_cols // 20)  # show at most 20 labels
        ax2.set_xticks(x_positions[::step])

    # Overall layout
    plt.tight_layout()
    # Save the figure
    if output_file:
        fig.savefig(output_file, format='jpg', dpi=200, bbox_inches='tight')
        print(f"✅ Heatmap and bar plot saved to: {output_file}")

    # plt.show()
    plt.close(fig)

    # Save binary matrix to Excel file (same name as output image, different suffix)
    base_name = os.path.splitext(output_file)[0]  # remove original file suffix
    excel_file = f"{base_name}.xlsx"

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        df = pd.DataFrame(compressed_data)
        df.to_excel(writer, sheet_name='activation', index=False, header=False)

    print(f"Binary matrix saved to {excel_file}")


def process_and_plot_tasks(task_list, compression_ratio, output_file='output.png'):
    """
    Process a list of task matrices and visualize the combined result (adaptive to number of tasks)
    """
    # 检查输入
    if len(task_list) == 0:
        raise ValueError("The task list cannot be empty")

    original_shape = task_list[0].shape
    for i, matrix in enumerate(task_list):
        assert matrix.shape == original_shape, f"Matrix {i} has incorrect shape, expected {original_shape}"

    num_tasks = len(task_list)

    # Step 1: Square all values
    squared_matrices = [np.square(matrix) for matrix in task_list]

    # Step 2: Compress each matrix by averaging over ratio x ratio blocks
    compressed_height = task_list[0].shape[0] // compression_ratio
    compressed_width = task_list[0].shape[1] // compression_ratio
    compressed_matrices = []

    for matrix in squared_matrices:
        compressed = np.zeros((compressed_height, compressed_width))
        for i in range(compressed_height):
            for j in range(compressed_width):
                block = matrix[i * compression_ratio:(i + 1) * compression_ratio, j * compression_ratio:(j + 1) * compression_ratio]
                compressed[i, j] = np.mean(block)
        compressed_matrices.append(compressed)

    # Step 3: Binarization (top 10% set to 1, others to 0)
    binary_matrices = []  # Original binary matrices: >=90th percentile set to 1, else 0
    normalized_matrices = []  # New: Min-Max normalize only the top 10% values, others set to 0

    for matrix in compressed_matrices:
        threshold_value = np.percentile(matrix, 90)  # 90th percentile as threshold

        # 1. Original binarization operation (unchanged)
        binary_matrix = (matrix >= threshold_value).astype(int)
        binary_matrices.append(binary_matrix)

        # 2. New: Min-Max normalize only the top 10% values, others set to 0
        top_10_mask = matrix >= threshold_value
        normalized_matrix = np.zeros_like(matrix, dtype=float)

        top_values = matrix[top_10_mask]
        if len(top_values) > 0:
            min_val, max_val = top_values.min(), top_values.max()
            if max_val > min_val:
                normalized_matrix[top_10_mask] = (top_values - min_val) / (max_val - min_val)
            else:
                normalized_matrix[top_10_mask] = 0.0  # all values are the same, normalize to 0

        normalized_matrices.append(normalized_matrix)

    # Compute the row sums after combining all tasks
    # Sum all binary matrices, then compute row sums
    combined_binary = np.sum(binary_matrices, axis=0)  # shape: (compressed_height, compressed_width)
    row_sums = np.sum(combined_binary, axis=1)  # sum for each row, shape: (compressed_height,)

    # Step 4: Visualization of overlay (using different colors, 20% opacity), with horizontal bar chart on the left
    fig = plt.figure(figsize=(18, 10))

    # Create grid layout: left bar chart takes 1 part, main plot takes 4 parts
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0.05)

    # Left: Horizontal bar chart (actually vertical bar chart, but represents row data)
    ax1 = fig.add_subplot(gs[0])
    y_pos = np.arange(compressed_height)
    bars = ax1.barh(y_pos, row_sums, height=0.8, color='steelblue', alpha=0.7)
    ax1.set_ylim(-0.5, compressed_height - 0.5)
    ax1.set_xlabel('Row Sum (Total Activations)')
    ax1.set_ylabel('Row Index')
    ax1.set_title('Row-wise Sum\nof All Tasks')
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Invert y-axis so that top of the image corresponds to first row of matrix
    ax1.invert_yaxis()

    # Main plot: Visualization of overlay result
    ax2 = fig.add_subplot(gs[1])

    # Choose appropriate color palette based on number of tasks
    if num_tasks <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:num_tasks]
    elif num_tasks <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        indices = np.linspace(0, 19, num_tasks, dtype=int)
        colors = colors[indices]
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, num_tasks))

    # Create a blank RGB image
    h, w = compressed_matrices[0].shape
    rgb_image = np.ones((h, w, 3))  # white background

    # Overlay binary matrices one by one
    for idx, binary_matrix in enumerate(binary_matrices):
        color = colors[idx % len(colors)][:3]  # get RGB values

        # For each position with 1, overlay the color onto the image with 20% opacity
        mask = binary_matrix == 1
        # Transparent overlay: final color = (original color × 0.8 + new color × 0.2)
        rgb_image[mask] = rgb_image[mask] * 0.8 + color * 0.2

    # Display the main image
    ax2.imshow(rgb_image)
    ax2.set_title(f'Combined Tasks Visualization ({num_tasks} tasks)')
    ax2.axis('off')

    # Adjust layout and save
    plt.savefig(output_file, dpi=600, bbox_inches='tight', format='png')
    plt.close()

    print(f"figure has been saved in {output_file}")

    # Save binary_matrices to Excel file (same name as output image, different suffix)
    base_name = os.path.splitext(output_file)[0]  # remove original file suffix
    excel_file = f"{base_name}.xlsx"

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for idx, binary_mat in enumerate(binary_matrices):
            # Convert numpy array to DataFrame
            df = pd.DataFrame(binary_mat)
            # Write to separate Excel sheets, named Task_0, Task_1, ...
            sheet_name = f"Task_{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    print(f"Binary matrices have been saved to {excel_file}")

    # Save binary_matrices to Excel file (same name as output image, different suffix)
    base_name = os.path.splitext(output_file)[0]  # remove original file suffix
    excel_file = f"{base_name}_norm.xlsx"

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        for idx, normal_mat in enumerate(normalized_matrices):
            # Convert numpy array to DataFrame
            df = pd.DataFrame(normal_mat)
            # Write to separate Excel sheets, named Task_0, Task_1, ...
            sheet_name = f"Task_{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)

    print(f"Normalized matrices have been saved to {excel_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--percent', type=int, default=10)
    parser.add_argument('--analysis_activation', type=bool, default=False)
    parser.add_argument('--analysis_grad', type=bool, default=False)
    parser.add_argument('--analysis_qkv_grad', type=bool, default=True)
    parser.add_argument('--compression_ratio', type=int, default=32)
    parser.add_argument('--root_path', type=str, default='analysis/')
    parser.add_argument('--analysis_name', type=str, default='llama3-3b-analysis-1025')
    args = parser.parse_args()

    analysis_path = Path(os.path.join(args.root_path, args.analysis_name))
    task_to_path = {}
    for subdir in analysis_path.iterdir():
        if not subdir.is_dir():
            continue
        dir_name = subdir.name  # e.g., avg_grads_task875_emotion_classification_llama3.2-3b-instruct

        # Try to match each task name in tasks
        matched = False
        for task in tasks:
            if task in dir_name:
                task_to_path[task] = str(subdir.name)  # save absolute path
                break  # one folder corresponds to one task

    layers = ['model.layers.1', 'model.layers.14', 'model.layers.27']  # llama3-3b
    layers = ['model.layers.1', 'model.layers.18', 'model.layers.35']  # qwen3-4b
    if 'llama' in args.analysis_name:
        low, medium, high = 1, 14, 27
    else:
        low, medium, high = 1, 18, 35

    if args.analysis_activation:
        print(f"***** Analyzing MLP ACTIVATION (gate)! *****")
        layer_low_act, layer_medium_act, layer_high_act = [], [], []
        for task in tqdm(task_to_path):
            mlp_activation_paths = os.path.join(analysis_path, task_to_path[task], f'mlp_gate_activation.pkl')
            mlp_gate_activation = joblib.load(mlp_activation_paths)

            layer_low_act.append(mlp_gate_activation[f"model.layers.{low}.mlp.gate_proj"])
            layer_medium_act.append(mlp_gate_activation[f"model.layers.{medium}.mlp.gate_proj"])
            layer_high_act.append(mlp_gate_activation[f"model.layers.{high}.mlp.gate_proj"])

        output_file = os.path.join(analysis_path, f'mlp_gate_act_keep{args.percent}_binary_low_{len(layer_low_act)}+{int(len(layer_low_act[0]) / args.compression_ratio)}.png')
        plot_heatmap_with_compression(layer_low_act, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'mlp_gate_act_keep{args.percent}_binary_medium_{len(layer_medium_act)}+{int(len(layer_medium_act[0]) / args.compression_ratio)}.png')
        plot_heatmap_with_compression(layer_medium_act, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'mlp_gate_act_keep{args.percent}_binary_high_{len(layer_high_act)}+{int(len(layer_high_act[0]) / args.compression_ratio)}.png')
        plot_heatmap_with_compression(layer_high_act, compression_ratio=args.compression_ratio, output_file=output_file)

    if args.analysis_grad:
        print(f"***** Analyzing MLP GRADIENT (gate)! *****")
        layer_low_grad, layer_medium_grad, layer_high_grad = [], [], []
        for task in tqdm(task_to_path):
            mlp_grad_paths = os.path.join(analysis_path, task_to_path[task], f'mlp_gate_grad.pkl')
            mlp_gate_grad = joblib.load(mlp_grad_paths)

            layer_low_grad.append(mlp_gate_grad[f"model.layers.{low}.mlp.gate_proj.weight"])
            layer_medium_grad.append(mlp_gate_grad[f"model.layers.{medium}.mlp.gate_proj.weight"])
            layer_high_grad.append(mlp_gate_grad[f"model.layers.{high}.mlp.gate_proj.weight"])

        output_file = os.path.join(analysis_path, f'mlp_gate_grad_keep{args.percent}_binary_low_(1-15)_{int(len(layer_low_grad[0]) / args.compression_ratio)}+{int(len(layer_low_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(layer_low_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'mlp_gate_grad_keep{args.percent}_binary_medium_(1-15)_{int(len(layer_low_grad[0]) / args.compression_ratio)}+{int(len(layer_low_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(layer_medium_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'mlp_gate_grad_keep{args.percent}_binary_high_(1-15)_{int(len(layer_low_grad[0]) / args.compression_ratio)}+{int(len(layer_low_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(layer_high_grad, compression_ratio=args.compression_ratio, output_file=output_file)

    if args.analysis_qkv_grad:
        print(f"***** Analyzing QKV GRADIENT (gate)! *****")
        q_layer_low_grad, q_layer_medium_grad, q_layer_high_grad = [], [], []
        k_layer_low_grad, k_layer_medium_grad, k_layer_high_grad = [], [], []
        v_layer_low_grad, v_layer_medium_grad, v_layer_high_grad = [], [], []
        for task in tqdm(task_to_path):
            q_grad_paths = os.path.join(analysis_path, task_to_path[task], f'q_gate_grad.pkl')
            q_grad = joblib.load(q_grad_paths)
            k_grad_paths = os.path.join(analysis_path, task_to_path[task], f'k_gate_grad.pkl')
            k_grad = joblib.load(k_grad_paths)
            v_grad_paths = os.path.join(analysis_path, task_to_path[task], f'v_gate_grad.pkl')
            v_grad = joblib.load(v_grad_paths)

            q_layer_low_grad.append(q_grad[f"model.layers.{low}.self_attn.q_proj.weight"])
            q_layer_medium_grad.append(q_grad[f"model.layers.{medium}.self_attn.q_proj.weight"])
            q_layer_high_grad.append(q_grad[f"model.layers.{high}.self_attn.q_proj.weight"])

            k_layer_low_grad.append(k_grad[f"model.layers.{low}.self_attn.k_proj.weight"])
            k_layer_medium_grad.append(k_grad[f"model.layers.{medium}.self_attn.k_proj.weight"])
            k_layer_high_grad.append(k_grad[f"model.layers.{high}.self_attn.k_proj.weight"])

            v_layer_low_grad.append(v_grad[f"model.layers.{low}.self_attn.v_proj.weight"])
            v_layer_medium_grad.append(v_grad[f"model.layers.{medium}.self_attn.v_proj.weight"])
            v_layer_high_grad.append(v_grad[f"model.layers.{high}.self_attn.v_proj.weight"])

        output_file = os.path.join(analysis_path, f'q_grad_keep{args.percent}_binary_low_(1-15)_{int(len(q_layer_low_grad[0]) / args.compression_ratio)}+{int(len(q_layer_low_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(q_layer_low_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'q_grad_keep{args.percent}_binary_medium_(1-15)_{int(len(q_layer_medium_grad[0]) / args.compression_ratio)}+{int(len(q_layer_medium_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(q_layer_medium_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'q_grad_keep{args.percent}_binary_high_(1-15)_{int(len(q_layer_high_grad[0]) / args.compression_ratio)}+{int(len(q_layer_high_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(q_layer_high_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        # --- *** >>>
        output_file = os.path.join(analysis_path, f'k_grad_keep{args.percent}_binary_low_(1-15)_{int(len(k_layer_low_grad[0]) / args.compression_ratio)}+{int(len(k_layer_low_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(k_layer_low_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'k_grad_keep{args.percent}_binary_medium_(1-15)_{int(len(k_layer_medium_grad[0]) / args.compression_ratio)}+{int(len(k_layer_medium_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(k_layer_medium_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'k_grad_keep{args.percent}_binary_high_(1-15)_{int(len(k_layer_high_grad[0]) / args.compression_ratio)}+{int(len(k_layer_high_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(k_layer_high_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        # --- *** >>>
        output_file = os.path.join(analysis_path, f'v_grad_keep{args.percent}_binary_low_(1-15)_{int(len(v_layer_low_grad[0]) / args.compression_ratio)}+{int(len(v_layer_low_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(v_layer_low_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'v_grad_keep{args.percent}_binary_medium_(1-15)_{int(len(v_layer_medium_grad[0]) / args.compression_ratio)}+{int(len(v_layer_medium_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(v_layer_medium_grad, compression_ratio=args.compression_ratio, output_file=output_file)

        output_file = os.path.join(analysis_path, f'v_grad_keep{args.percent}_binary_high_(1-15)_{int(len(v_layer_high_grad[0]) / args.compression_ratio)}+{int(len(v_layer_high_grad[0][0]) / args.compression_ratio)}.png')
        process_and_plot_tasks(v_layer_high_grad, compression_ratio=args.compression_ratio, output_file=output_file)
