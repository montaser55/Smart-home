import pandas as pd
import numpy as np
import ast
import re
import argparse


def parse_flow_string(flow_string):
    """
    Convert something like:
      [[receive,61],[receive,69],[send,51]]
    into a valid Python list by inserting quotes around direction words
    and using ast.literal_eval.
    """
    direction_words = ['receive', 'incoming', 'send', 'outgoing']
    pattern = r'(\W)(' + '|'.join(direction_words) + r')(\W)'
    replacement = r'\1"\2"\3'

    corrected = re.sub(pattern, replacement, flow_string)
    return ast.literal_eval(corrected)


def compute_stats_and_abs_cumulative_sums(flow_pairs):
    """
    1) Count # of 'receive/incoming' vs 'send/outgoing'
    2) Sum their sizes
    3) Build an "absolute" cumulative sum (always add size, ignoring direction sign).
    """
    incoming_packets = 0
    outgoing_packets = 0
    incoming_size_sum = 0
    outgoing_size_sum = 0

    cumulative_sums = []
    running_sum = 0

    for direction_str, size in flow_pairs:
        direction = direction_str.lower().strip()

        # For stats only:
        if direction in ['receive', 'incoming']:
            incoming_packets += 1
            incoming_size_sum += size
        else:
            outgoing_packets += 1
            outgoing_size_sum += size

        # "Absolute sum" approach = always add
        running_sum += size
        cumulative_sums.append(running_sum)

    stats = {
        'incoming_packets': incoming_packets,
        'outgoing_packets': outgoing_packets,
        'incoming_size_sum': incoming_size_sum,
        'outgoing_size_sum': outgoing_size_sum
    }
    return stats, cumulative_sums


def sample_cumulative_sums(cumulative_sums, m):
    """
    Down-sample cumulative_sums to m points using integer indices.
    NOTE: If m > len(cumulative_sums), you may see repeated values!
    """
    if not cumulative_sums:
        return [0] * m

    indices = np.linspace(0, len(cumulative_sums) - 1, m).astype(int)
    return [cumulative_sums[i] for i in indices]


def process_flows(input_csv, output_csv, m):
    """
    1) Skip first line of CSV.
    2) Read each flow line as one column "flow".
    3) For each flow, parse + compute absolute cumulative sums.
    4) Sample sums at m points -> repeated values if m > #packets!
    5) Output a CSV row per flow with stats + sums.
    """
    df_in = pd.read_csv(
        input_csv,
        sep='|',
        engine='python',
        header=None,
        names=['flow'],
        skiprows=1  # skip the header "Packet Size and Direction"
    )

    results = []
    for _, row in df_in.iterrows():
        flow_string = str(row['flow']).strip()
        if not flow_string:
            continue

        # Parse the flow (insert quotes, then literal_eval)
        flow_pairs = parse_flow_string(flow_string)

        # Compute stats and absolute cumulative sums
        stats, abs_sums = compute_stats_and_abs_cumulative_sums(flow_pairs)

        # Sample those sums at m points
        sampled = sample_cumulative_sums(abs_sums, m)

        # Build result row
        row_dict = {
            'incoming_packets': stats['incoming_packets'],
            'outgoing_packets': stats['outgoing_packets'],
            'incoming_size_sum': stats['incoming_size_sum'],
            'outgoing_size_sum': stats['outgoing_size_sum']
        }
        for i, val in enumerate(sampled, start=1):
            row_dict[f'CumulativeSum_{i}'] = val

        results.append(row_dict)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"[INFO] Processed {len(df_out)} flows => {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Absolute cum sum of packet sizes (ignore direction sign).")
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--m', type=int, required=True,
                        help="Number of sample points. Be careful if m is larger than # packets!")
    args = parser.parse_args()

    process_flows(args.input_csv, args.output_csv, args.m)


if __name__ == "__main__":
    main()
