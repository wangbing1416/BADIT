# python
import re
import argparse
from collections import defaultdict
import math
import numpy as np
import pandas as pd


def parse_log_to_excel_by_layer(log_content):
    # Initialize data structure
    epoch_layer_data = {}

    # Split log by lines
    lines = log_content.strip().split('\n')

    # Track current cycle and epoch
    current_cycle = 1  # Start counting from 1
    previous_epoch = 0  # For detecting the start of a new cycle
    current_epoch = None

    # Traverse the log to find the cos value for each epoch
    for i in range(len(lines)):
        line = lines[i].strip()

        # Check if it is an EPOCH line
        epoch_match = re.search(r'>> EPOCH: (\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

            # Check if it is the start of a new cycle
            if current_epoch < previous_epoch:
                current_cycle += 1

            previous_epoch = current_epoch
            continue

        # Check if the line contains cos values and layer info
        intra_match = re.search(r'Intra-expert cos: ([+-]?\d+\.\d+)', line)
        inter_match = re.search(r'Inter-expert cos: ([+-]?\d+\.\d+)', line)
        layer_match = re.search(r'model\.layers\.(\d+)\.mlp\.gate_proj', line)

        if intra_match and inter_match and layer_match and current_epoch is not None:
            intra_cos = float(intra_match.group(1))
            inter_cos = float(inter_match.group(1))
            layer_num = int(layer_match.group(1))

            # Use the composite key of current cycle and epoch
            key = (current_cycle, current_epoch)

            # Initialize nested dictionary
            if key not in epoch_layer_data:
                epoch_layer_data[key] = {}
            if layer_num not in epoch_layer_data[key]:
                epoch_layer_data[key][layer_num] = {'intra_cos': [], 'inter_cos': []}

            epoch_layer_data[key][layer_num]['intra_cos'].append(intra_cos)
            epoch_layer_data[key][layer_num]['inter_cos'].append(inter_cos)

    # Compute the average value for each layer in each epoch of each cycle and convert to angle
    results = []
    for (cycle, epoch), layers_data in epoch_layer_data.items():
        for layer, data in layers_data.items():
            if data['intra_cos'] and data['inter_cos']:
                avg_intra_cos = sum(data['intra_cos']) / len(data['intra_cos'])
                avg_inter_cos = sum(data['inter_cos']) / len(data['inter_cos'])

                # Convert cos value to angle
                intra_angle = math.degrees(math.acos(max(-1, min(1, avg_intra_cos))))
                inter_angle = math.degrees(math.acos(max(-1, min(1, avg_inter_cos))))

                results.append({
                    'Cycle': cycle,
                    'Epoch': epoch,
                    'Layer': layer,
                    'Intra-expert Angle': intra_angle,
                    'Inter-expert Angle': inter_angle
                })

    # Create DataFrame
    df = pd.DataFrame(results, columns=['Cycle', 'Epoch', 'Layer', 'Intra-expert Angle', 'Inter-expert Angle'])

    # Sort by Cycle, Epoch, and Layer
    df = df.sort_values(by=['Cycle', 'Epoch', 'Layer']).reset_index(drop=True)

    return df


def parse_log_to_excel(log_content):
    # Initialize data structure
    epoch_data = defaultdict(lambda: {'intra_cos': [], 'inter_cos': []})

    # Split log by lines
    lines = log_content.strip().split('\n')

    # Track current cycle and epoch
    current_cycle = 1  # Start counting from 1
    previous_epoch = 1  # For detecting the start of a new cycle
    current_epoch = None

    # Traverse the log to find the cos value for each epoch
    for i in range(len(lines)):
        line = lines[i].strip()

        # Check if it is an EPOCH line
        epoch_match = re.search(r'>> EPOCH: (\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

            # Check if it is the start of a new cycle
            if current_epoch < previous_epoch:
                current_cycle += 1

            previous_epoch = current_epoch
            continue

        # Check if the line contains cos values
        intra_match = re.search(r'Intra-expert cos: ([+-]?\d+\.\d+)', line)
        inter_match = re.search(r'Inter-expert cos: ([+-]?\d+\.\d+)', line)

        if intra_match and inter_match and current_epoch is not None:
            intra_cos = float(intra_match.group(1))
            inter_cos = float(inter_match.group(1))

            # Use the composite key of current cycle and epoch
            key = (current_cycle, current_epoch)
            epoch_data[key]['intra_cos'].append(intra_cos)
            epoch_data[key]['inter_cos'].append(inter_cos)

    # Compute the average value for each epoch and convert to angle
    results = []
    for (cycle, epoch), data in epoch_data.items():
        if data['intra_cos'] and data['inter_cos']:
            avg_intra_cos = sum(data['intra_cos']) / len(data['intra_cos'])
            avg_inter_cos = sum(data['inter_cos']) / len(data['inter_cos'])

            # Convert cos value to angle
            intra_angle = math.degrees(math.acos(max(-1, min(1, avg_intra_cos))))
            inter_angle = math.degrees(math.acos(max(-1, min(1, avg_inter_cos))))

            results.append({
                'Cycle': cycle,
                'Epoch': epoch,
                'Intra-expert Angle': intra_angle,
                'Inter-expert Angle': inter_angle
            })

    # Create DataFrame
    df = pd.DataFrame(results, columns=['Cycle', 'Epoch', 'Intra-expert Angle', 'Inter-expert Angle'])

    # Sort by Cycle and Epoch
    df = df.sort_values(by=['Cycle', 'Epoch']).reset_index(drop=True)

    return df


def parse_log_file_to_excel(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        log_content = file.read()

    df = parse_log_to_excel(log_content)
    # Save to Excel file
    output_file = file_path.replace('.log', '_angles.xlsx')
    df.to_excel(output_file, index=False)
    print(f"Results saved to: {output_file}")

    # by layer
    df_layer = parse_log_to_excel_by_layer(log_content)
    output_file_layer = file_path.replace('.log', '_angles_by_layer.xlsx')
    df_layer.to_excel(output_file_layer, index=False)
    print(f"Layer-wise results saved to: {output_file_layer}")

    return df

def main():
    parser = argparse.ArgumentParser(description="Parse log and compute intra/inter angles per epoch")
    parser.add_argument("--name", default="qwen3-8b-loramoe_group-seed1")
    args = parser.parse_args()

    logfile = f"output/{args.name}/regroup_logger.log"
    _ = parse_log_file_to_excel(logfile)

if __name__ == "__main__":
    main()
