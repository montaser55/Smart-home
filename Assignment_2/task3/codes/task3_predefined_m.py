import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


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


def generate_feature_vectors(cumulative_df, m_values, fp):
    output_directory = "../output/feature_vectors"
    os.makedirs(output_directory, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(fp))[0]

    for m in m_values:
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        output_file_name = f"{base_name}_feature_vector_m_{m}.csv"
        output_file_path = f"{output_directory}/{output_file_name}"

        sampled_features.to_csv(output_file_path, index=False)
        print(f"Feature vector for m={m} saved to {output_file_path}")


def plot_feature_vectors_side_by_side(cumulative_df, m_values, fp):
    output_directory = "../output/feature_vector_plots"
    os.makedirs(output_directory, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(fp))[0]

    num_subplots = len(m_values)
    fig, axes = plt.subplots(1, num_subplots, figsize=(8 * num_subplots, 6), sharey=True)

    if num_subplots == 1:
        axes = [axes]

    for i, (ax, m) in enumerate(zip(axes, m_values)):
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        ax.plot(sampled_features["AbsoluteSum"], sampled_features["CumulativeSum"], marker=".", markersize=4, label=f"m={m}")

        ax.set_title(f"m={m}")
        ax.set_xlabel("Absolute Sum")
        ax.set_ylabel("Cumulative Sum")
        ax.legend()

    plt.suptitle("Feature Vectors for Communication Flow", fontsize=16)

    output_file_path = f"{output_directory}/{base_name}_feature_vectors_side_by_side.png"
    plt.savefig(output_file_path)
    print(f"Plot saved to {output_file_path}")

    plt.show()


# Base Directory and File Names
base_directory = "../dataset/packet_size_and_direction"
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
file_paths = [f"{base_directory}/{file_name}" for file_name in file_names]

# Argument Parsing
parser = argparse.ArgumentParser(description="Process and visualize packet size and direction data.")
parser.add_argument('--list', action='store_true', help="List all available files with their indices.")
parser.add_argument('--file_index', type=int, help="Index of the file to process (1-based index).")

args = parser.parse_args()

# Handle --list
if args.list:
    print("\nAvailable Files:")
    for idx, file_name in enumerate(file_names, start=1):
        print(f"{idx}: {file_name}")
    exit()

# Validate File Index
if args.file_index is None:
    print("Error: --file_index is required unless using --list.")
    parser.print_help()
    exit()

selected_index = args.file_index - 1
if 0 <= selected_index < len(file_paths):
    file_path = file_paths[selected_index]
    print(f"\nSelected file: {file_path}")

    # Predefined m values
    m_values = [90, 150, 200]

    # Load and process data
    df = load_and_add_data(file_path)
    calculate_statistics(df)
    cumulative_df = compute_cumulative_representation(df)
    generate_feature_vectors(cumulative_df, m_values, file_path)
    plot_feature_vectors_side_by_side(cumulative_df, m_values, file_path)
else:
    print("Invalid file index. Use --list to see available files.")
    exit()
