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


def generate_feature_vectors(cumulative_df, m_values):
    for m in m_values:

        # Sample m equidistant points
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        # Save each sampled feature vector to its own CSV file
        sampled_features.to_csv(f"feature_vector_m_{m}.csv")
        print(f"Feature vector for m={m} saved to feature_vector_m_{m}.csv")


def plot_feature_vectors(cumulative_df, m_values):
    plt.figure(figsize=(10, 6))

    for m in m_values:
        indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
        sampled_features = cumulative_df.iloc[indices]

        plt.plot(sampled_features["AbsoluteSum"], sampled_features["CumulativeSum"], marker="o", markersize=4, label=f"m={m}")

    plt.title("Feature Vectors for Communication Flow")
    plt.xlabel("Absolute Sum")
    plt.ylabel("Cumulative Sum")
    plt.legend()
    plt.show()

# main
file_path = "output_file2.csv"
m_values = [90, 150, 200]
df = load_and_add_data(file_path)
calculate_statistics(df)
print(f"\n {df}")
cumulative_df = compute_cumulative_representation(df)
print(f"\n {cumulative_df}")
generate_feature_vectors(cumulative_df, m_values)
plot_feature_vectors(cumulative_df, m_values)
