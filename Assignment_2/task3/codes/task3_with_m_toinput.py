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


def plot_feature_vector(cumulative_df, m):
    indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
    sampled_features = cumulative_df.iloc[indices]

    plt.figure(figsize=(8, 6))
    plt.plot(sampled_features["AbsoluteSum"], sampled_features["CumulativeSum"], marker=".", markersize=4, label=f"m={m}")

    plt.title(f"Feature Vector for m={m}")
    plt.xlabel("Absolute Sum")
    plt.ylabel("Cumulative Sum")
    plt.legend()
    plt.show()


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

print("Available Files:")
for idx, file_path in enumerate(file_names, start=1):
    print(f"{idx}: {file_path}")

selected_index = int(input("Enter the number corresponding to the file you want to process: ").strip()) - 1

if 0 <= selected_index < len(file_paths):
    file_path = file_paths[selected_index]
    print(f"\nSelected file: {file_path}")

    df = load_and_add_data(file_path)
    calculate_statistics(df)
    cumulative_df = compute_cumulative_representation(df)

    m = int(input("Enter the value for m (number of sampled points): ").strip())
    plot_feature_vector(cumulative_df, m)
else:
    print("Invalid selection. Please try again.")
