import numpy as np

# Example data: Packet lengths
packet_lengths = [100, 200, 150, 300, 200]

# Step 1: Sort the packet lengths
sorted_lengths = np.sort(packet_lengths)  # Output: [100, 150, 200, 200, 300]

# Step 2: Compute CDF values
print(np.arange(1, len(sorted_lengths) + 1))
print(len(sorted_lengths))
cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
# Output: [0.2, 0.4, 0.6, 0.8, 1.0]
print(sorted_lengths)
print(cdf)
# Visualizing the CDF
import matplotlib.pyplot as plt
plt.step(sorted_lengths, cdf, where='post')
plt.xlabel('Packet Size')
plt.ylabel('CDF')
plt.title('CDF of Packet Sizes')
plt.grid()
plt.show()