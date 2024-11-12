import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("output_file.csv")  # Replace with your actual file path

# Step 1: Calculate Incoming and Outgoing Packet Counts
incoming_packets = df[df["Direction"] == "incoming"].shape[0]
outgoing_packets = df[df["Direction"] == "outgoing"].shape[0]
print("Incoming packets:", incoming_packets)
print("Outgoing packets:", outgoing_packets)

# Step 2: Calculate Sum of Incoming and Outgoing Packet Sizes
incoming_packet_size_sum = df[df["Direction"] == "incoming"]["PacketSize"].sum()
outgoing_packet_size_sum = df[df["Direction"] == "outgoing"]["PacketSize"].sum()
print("Sum of incoming packet sizes:", incoming_packet_size_sum)
print("Sum of outgoing packet sizes:", outgoing_packet_size_sum)

# Step 3: Generate Cumulative Representation of Packet Sizes
cumulative_packets = [(0, 0)]
cumulative_sum = 0
absolute_sum = 0

for size in df["PacketSize"]:
    cumulative_sum += size
    absolute_sum += abs(size)
    cumulative_packets.append((absolute_sum, cumulative_sum))

# Convert to DataFrame for cumulative data
cumulative_df = pd.DataFrame(cumulative_packets, columns=["AbsoluteSum", "CumulativeSum"])
print(cumulative_df.head(10))

# Step 4: Sample the Cumulative Data at m Equidistant Points
m = 90  # Define the number of points for sampling; adjust as needed (e.g., 90, 150, 200)
indices = np.linspace(0, len(cumulative_df) - 1, m).astype(int)
sampled_features = cumulative_df.iloc[indices]

print("Sampled Features:\n", sampled_features)

# # Step 5: Save or Plot the Feature Vector
# # Save the feature vector (optional)
# sampled_features.to_csv(f"feature_vector_m_{m}.csv", index=False)
#
# # Plot the feature vector
# plt.plot(sampled_features["AbsoluteSum"], sampled_features["CumulativeSum"], marker="o")
# plt.title(f"Feature Vector for m = {m}")
# plt.xlabel("Absolute Sum")
# plt.ylabel("Cumulative Sum")
# plt.show()
