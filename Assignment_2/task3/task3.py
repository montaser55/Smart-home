import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def generate_feature_vectors(cumulative_df, m_values, input_file_name):
    # Ensure the output directory exists
    output_directory = "../output/feature_vectors"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract the base name without the extension
    base_name = os.path.splitext(os.path.basename(input_file_name))[0]

    for m in m_values:
        # Sample m equidistant points
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        # Construct the output file name
        output_file_name = f"{base_name}_feature_vector_m_{m}.csv"
        output_file_path = os.path.join(output_directory, output_file_name)

        # Save the sampled features to the output directory
        sampled_features.to_csv(output_file_path, index=False)
        print(f"Feature vector for m={m} saved to {output_file_path}")


def plot_feature_vectors(cumulative_df, m_values, input_file_name):
    # Ensure the output directory exists
    output_directory = "../output/plots/feature_vector_plots"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract the base name without the extension
    base_name = os.path.splitext(os.path.basename(input_file_name))[0]

    plt.figure(figsize=(10, 6))

    for m in m_values:
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        plt.plot(sampled_features["AbsoluteSum"], sampled_features["CumulativeSum"], linestyle='-', linewidth=1, label=f"m={m}")

    plt.title("Feature Vectors for Communication Flow")
    plt.xlabel("Absolute Sum")
    plt.ylabel("Cumulative Sum")
    plt.legend()

    # Save the plot to the output directory
    output_file_path = os.path.join(output_directory, f"{base_name}_feature_vectors_plot.png")
    plt.savefig(output_file_path)
    print(f"Plot saved to {output_file_path}")

    plt.show()


def plot_feature_vectors_with_subplots(cumulative_df, m_values, input_file_name):
    # Ensure the output directory exists
    output_directory = "../output/plots/feature_vector_plots"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract the base name without the extension
    base_name = os.path.splitext(os.path.basename(input_file_name))[0]

    # Create subplots for each m value
    num_subplots = len(m_values)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6 * num_subplots), sharex=True)

    if num_subplots == 1:
        axes = [axes]  # Ensure axes is always iterable

    for idx, (ax, m) in enumerate(zip(axes, m_values)):
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        # Plot data on the current subplot
        ax.plot(
            sampled_features["AbsoluteSum"],
            sampled_features["CumulativeSum"],
            marker=".",
            markersize=4,
            label=f"m={m}"
        )

        ax.set_title(f"Feature Vectors for Communication Flow (m={m})")
        ax.set_xlabel("Absolute Sum")
        ax.set_ylabel("Cumulative Sum")
        ax.legend()

    plt.tight_layout()

    # Save the plot to the output directory
    output_file_path = os.path.join(output_directory, f"{base_name}_feature_vectors_subplots.png")
    plt.savefig(output_file_path)
    print(f"Plot saved to {output_file_path}")

    plt.show()


def plot_feature_vectors_side_by_side(cumulative_df, m_values, input_file_name):
    # Ensure the output directory exists
    output_directory = "../output/plots/feature_vector_plots"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Extract the base name without the extension
    base_name = os.path.splitext(os.path.basename(input_file_name))[0]

    # Create subplots for each m value (side by side)
    num_subplots = len(m_values)
    fig, axes = plt.subplots(1, num_subplots, figsize=(8 * num_subplots, 6), sharey=True)

    if num_subplots == 1:
        axes = [axes]  # Ensure axes is always iterable

    for idx, (ax, m) in enumerate(zip(axes, m_values)):
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        # Plot data on the current subplot
        ax.plot(
            sampled_features["AbsoluteSum"],
            sampled_features["CumulativeSum"],
            marker=".",
            markersize=4,
            label=f"m={m}"
        )

        ax.set_title(f"m={m}")
        ax.set_xlabel("Absolute Sum")
        ax.set_ylabel("Cumulative Sum")
        ax.legend()

    plt.suptitle("Feature Vectors for Communication Flow", fontsize=16)
    plt.tight_layout()

    # Save the plot to the output directory
    output_file_path = os.path.join(output_directory, f"{base_name}_feature_vectors_side_by_side.png")
    plt.savefig(output_file_path)
    print(f"Plot saved to {output_file_path}")

    plt.show()


# Base directory for the files
base_directory = "../dataset/csv/packet_size_and_direction"

# File names extracted from the image
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

# Prepend the base directory to each file name
file_paths = [os.path.join(base_directory, file_name) for file_name in file_names]

# Display available files
print("Available Files:")
for idx, file_path in enumerate(file_names, start=1):
    print(f"{idx}: {file_path}")

# User selects a file
selected_index = int(input("Enter the number corresponding to the file you want to process: ").strip()) - 1

if 0 <= selected_index < len(file_paths):
    file_path = file_paths[selected_index]
    print(f"\nSelected file: {file_path}")

    # Load and process the selected file
    df = load_and_add_data(file_path)
    calculate_statistics(df)
    cumulative_df = compute_cumulative_representation(df)

    m_values = [90, 150, 200]
    generate_feature_vectors(cumulative_df, m_values, file_path)
    plot_feature_vectors_side_by_side(cumulative_df, m_values, file_path)
    # plot_feature_vectors_with_subplots(cumulative_df, m_values, file_path)
    # plot_feature_vectors(cumulative_df, m_values, file_path)
else:
    print("Invalid selection. Please try again.")
