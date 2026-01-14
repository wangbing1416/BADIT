import argparse
import re
import pandas as pd
from datetime import datetime


def process_training_log(input_file, output_file):
    """
    Process training log file, deduplicate consecutive duplicates, and export to Excel.

    Args:
        input_file: Path to the input log file.
        output_file: Path to the output Excel file.
    """
    # Read the log file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Regex to match the training log format
    # Matches: timestamp - [Epoch X] loss: Y, grad_norm: Z, learning_rate: A, epoch: B, step: C
    pattern = r'(\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}:\d{2}) - \[Epoch (\d+)\] loss: ([\d.]+|nan|inf), grad_norm: ([\d.e+-]+), learning_rate: ([\d.e+-]+), epoch: ([\d.]+), step: (\d+)'

    records = []

    for line in lines:
        line = line.strip()
        match = re.match(pattern, line)

        if match:
            timestamp = match.group(1)
            epoch = int(match.group(2))
            loss = float(match.group(3)) if match.group(3) not in ['nan', 'inf'] else match.group(3)
            grad_norm = float(match.group(4))
            learning_rate = float(match.group(5))
            epoch_float = float(match.group(6))
            step = int(match.group(7))

            records.append({
                'timestamp': timestamp,
                'epoch': epoch,
                'loss': loss,
                'grad_norm': grad_norm,
                'learning_rate': learning_rate,
                'epoch_float': epoch_float,
                'step': step
            })

    # Deduplicate consecutive identical records: keep only the first in each run
    unique_records = []
    i = 0
    while i < len(records):
        current_record = records[i]
        unique_records.append(current_record)

        # Check following entries for identical consecutive records
        j = i + 1
        while j < len(records) and \
                records[j]['epoch'] == current_record['epoch'] and \
                records[j]['loss'] == current_record['loss']:
            j += 1

        # Skip all duplicates and continue with the next different record
        i = j

    # Create DataFrame
    df = pd.DataFrame(unique_records)

    # Preserve original order (already chronological)
    df = df.reset_index(drop=True)

    # Write to Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Training_Logs', index=False)

        # Create a sheet containing only loss values
        loss_df = df[['epoch', 'loss', 'step']].copy()
        loss_df.to_excel(writer, sheet_name='Loss_Values', index=False)

    print(f"Processing complete!")
    print(f"Original record count: {len(records)}")
    print(f"Record count after deduplication: {len(df)}")
    print(f"Results saved to: {output_file}")

    # Display the first few and last few rows
    print("\nFirst 10 rows after deduplication:")
    print(df.head(10)[['epoch', 'loss', 'step']])

    print("\nLast 10 rows after deduplication:")
    print(df.tail(10)[['epoch', 'loss', 'step']])

    return df


def main():
    # Configure input and output file paths
    parser = argparse.ArgumentParser(description="Parse log and compute intra/inter angles per epoch")
    parser.add_argument("--name", default="gemma2-9b-loramoe_group-seed2")
    args = parser.parse_args()

    input_file = f'output/{args.name}/loss_logger.log'  # input log file
    output_file = f'output/{args.name}/cleaned_training_logs.xlsx'  # output Excel file

    # Process the log file
    df = process_training_log(input_file, output_file)

    # Statistics
    print(f"\nTraining statistics:")
    print(f"Total epochs: {df['epoch'].max()}")
    print(f"Total steps: {df['step'].max()}")
    print(f"Initial loss: {df.iloc[0]['loss']}")
    print(f"Final loss: {df.iloc[-1]['loss']}")
    print(f"Minimum loss: {df['loss'].min()}")

    # Records per epoch
    print("\nRecords per epoch:")
    epoch_counts = df['epoch'].value_counts().sort_index()
    for epoch, count in epoch_counts.items():
        print(f"Epoch {epoch}: {count} records")


if __name__ == "__main__":
    main()

