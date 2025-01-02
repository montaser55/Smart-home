import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


def adjust_packet_size(row):
    if row['Direction'] == 'outgoing':
        return row['PacketSize']
    else:
        return -row['PacketSize']


def load_and_add_data(file_path):
    df = pd.read_csv(file_path)
    df['AdjustedPacketSize'] = df.apply(adjust_packet_size, axis=1)
    return df


def calculate_statistics(df):
    incoming_packets = df[df['Direction'] == 'incoming'].shape[0]
    outgoing_packets = df[df['Direction'] == 'outgoing'].shape[0]
    incoming_size_sum = df[df['Direction'] == 'incoming']['PacketSize'].sum()
    outgoing_size_sum = df[df['Direction'] == 'outgoing']['PacketSize'].sum()

    print("Incoming packets:", incoming_packets)
    print("Outgoing packets:", outgoing_packets)
    print("Sum of incoming packet sizes:", incoming_size_sum)
    print("Sum of outgoing packet sizes:", outgoing_size_sum)


def compute_cumulative_representation(df):
    cumulative_sum = 0
    absolute_sum = 0
    cumulative_packets = [(0, 0)]

    for size in df['AdjustedPacketSize']:
        cumulative_sum += size
        absolute_sum += abs(size)
        cumulative_packets.append((absolute_sum, cumulative_sum))

    cumulative_df = pd.DataFrame(cumulative_packets, columns=["AbsoluteSum", "CumulativeSum"])
    return cumulative_df


def plot_feature_vector(cumulative_df, m):
    indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
    sampled_features = cumulative_df.iloc[indices]
    print(sampled_features)
    plt.figure(figsize=(8, 6))
    plt.plot(sampled_features["AbsoluteSum"], sampled_features["CumulativeSum"], marker=".", markersize=4, label=f"m={m}")

    plt.title(f"Feature Vector for m={m}")
    plt.xlabel("Absolute Sum")
    plt.ylabel("Cumulative Sum")
    plt.legend()
    plt.show()

def main():

    file_names = [
        "scenario1_doorsensor_to_coordinator.csv",
        "scenario1_ledvance_to_coordinator.csv",
        "scenario1_osarm_to_coordinator.csv",
        "scenario2_ledvance_to_coordinator.csv",
        "scenario2_osarm_to_coordinator.csv",
        "scenario2_waterleaksensor_to_coordinator.csv",
        "scenario3_doorsensor_to_coordinator.csv",
        "scenario3_ledvance_to_coordinator.csv",
        "scenario3_motionsensor_to_coordinator.csv",
        "scenario4_ledvance_to_coordinator.csv",
        "scenario4_motionsensor_to_coordinator.csv",
        "scenario4_osarm_to_coordinator.csv",
        "scenario5_ledvance_to_coordinator.csv",
        "scenario5_motionsensor_to_coordinator.csv",
        "scenario5_osarm_to_coordinator.csv",
        "scenario5_outdoormotionsensor_to_coordinator.csv",
        "scenario6_frientdoorsensor_to_coordinator.csv",
        "scenario6_ledvance_to_coordinator.csv",
        "scenario6_nedisdoorsensor_to_coordinator.csv",
        "scenario6_osarm_to_coordinator.csv"
    ]


    parser = argparse.ArgumentParser(description="Process and visualize packet size and direction data.")
    parser.add_argument('--base_directory', type=str, help="Base directory containing the CSV files.")
    parser.add_argument('--list', action='store_true', help="List all available files with their indices.")
    parser.add_argument('--file_index', type=int, help="Index of the file to process (1-based index).")
    parser.add_argument('--m', type=int, help="Value for m (number of sampled points).")

    args = parser.parse_args()

    file_paths = [f"{args.base_directory}/{file_name}" for file_name in file_names]

    if args.list:
        print("\nAvailable Files:")
        for idx, file_name in enumerate(file_names, start=1):
            print(f"{idx}: {file_name}")
        sys.exit(0)

    if args.file_index is None or args.m is None or args.base_directory is None:
        print("Error: --file_index, --m and --base_directory are required unless using --list.")
        parser.print_help()
        sys.exit(1)

    selected_index = args.file_index - 1
    if 0 <= selected_index < len(file_paths):
        file_path = file_paths[selected_index]
        print(f"\nSelected file: {file_path}")

        df = load_and_add_data(file_path)
        calculate_statistics(df)
        cumulative_df = compute_cumulative_representation(df)

        plot_feature_vector(cumulative_df, args.m)
    else:
        print("Invalid file index. Use --list to see available files.")
        sys.exit(1)

if __name__ == "__main__":
    main()