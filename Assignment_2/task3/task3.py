import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File path for your CSV file
file_path = "output_file.csv"  # Replace this with your actual file path

# Step 1: Load CSV data with direction and packet size
df = pd.read_csv(file_path)

# Define different values of m for sampling
m_values = [90, 150, 200]

# Step 2: Adjust packet sizes based on direction
df['AdjustedPacketSize'] = df.apply(lambda row: row['PacketSize'] if row['Direction'] == 'outgoing' else -row['PacketSize'], axis=1)

# Step 3: Calculate feature statistics for the communication flow
incoming_packets = df[df['Direction'] == 'incoming'].shape[0]
outgoing_packets = df[df['Direction'] == 'outgoing'].shape[0]
incoming_size_sum = df[df['Direction'] == 'incoming']['PacketSize'].sum()
outgoing_size_sum = df[df['Direction'] == 'outgoing']['PacketSize'].sum()

# Display flow statistics
print("Incoming packets:", incoming_packets)
print("Outgoing packets:", outgoing_packets)
print("Sum of incoming packet sizes:", incoming_size_sum)
print("Sum of outgoing packet sizes:", outgoing_size_sum)

# Step 4: Generate cumulative representation using AdjustedPacketSize
cumulative_sum = 0
absolute_sum = 0
cumulative_packets = [(0, 0)]  # Start with (0, 0) as per the task instructions

for size in df['AdjustedPacketSize']:
    cumulative_sum += size
    absolute_sum += abs(size)
    cumulative_packets.append((absolute_sum, cumulative_sum))

cumulative_df = pd.DataFrame(cumulative_packets, columns=["AbsoluteSum", "CumulativeSum"])
print(f"\n {cumulative_df}")

# Step 5: Plot feature vectors for different values of m
plt.figure(figsize=(10, 6))  # Create a single figure for all m values

for m in m_values:
    # Sample m equidistant points
    indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
    sampled_features = cumulative_df.iloc[indices]

    # Save each sampled feature vector to its own CSV file
    sampled_features.to_csv(f"feature_vector_m_{m}.csv", index=False)
    print(f"Feature vector for m={m} saved to feature_vector_m_{m}.csv")

    # Plot the feature vector for this value of m
    plt.plot(sampled_features["AbsoluteSum"], sampled_features["CumulativeSum"], marker="o", markersize=4, label=f"m={m}")

# Show plot with legend for different m values
plt.title("Feature Vectors for Communication Flow")
plt.xlabel("Absolute Sum")
plt.ylabel("Cumulative Sum")
plt.legend()
plt.show()
