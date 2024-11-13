import pyshark
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# file_path = 'scenario1.pcapng'
# device_addresses = {"0x7290", "0xcf6c", "0x8b7c", "0x0000"}

file_path = 'scenario2.pcapng'
device_addresses = {"0xe7fb", "0xcf6c", "0x8b7c", "0x0000"}

# file_path = 'scenario3.pcapng'
# motion_sensor_addresses = {"0xd0b9", "0xdc52", "0xb547", "0xacba", "0xdd43"}
# other_device_addresses = {"0xe6c4", "0xcf6c", "0x0000", "MotionSensor"}
# device_addresses = motion_sensor_addresses.union(other_device_addresses)

# file_path = 'scenario4.pcapng'
# device_addresses = {"0xd0b9", "0xcf6c", "0x8b7c", "0x0000"}

# file_path = 'scenario5.pcapng'
# device_addresses = {"0xdd43", "0x6ef9", "0xcf6c", "0x8b7c", "0x0000"}

# file_path = 'scenario6.pcapng'
# device_addresses = {"0x7290", "0xe6c4", "0xcf6c", "0x8b7c", "0x0000"}

packet_lengths = defaultdict(list)
inter_arrival_times = defaultdict(lambda: defaultdict(list))

print(f"Processing file: {file_path}")
capture = pyshark.FileCapture(file_path)

previous_timestamps = {}

for packet in capture:
    try:
        # take highest_layer for each packet and record its size
        protocol = packet.highest_layer
        packet_length = int(packet.length)
        packet_lengths[protocol].append(packet_length)

        if hasattr(packet, 'zbee_nwk'):
            src_addr = packet.zbee_nwk.src
            dst_addr = packet.zbee_nwk.dst

            # just for scenario 3
            # if src_addr in motion_sensor_addresses:
            #     src_addr = "MotionSensor"
            # if dst_addr in motion_sensor_addresses:
            #     dst_addr = "MotionSensor"

            # filter inter-arrival times on specified device addresses
            if (src_addr == "0x0000" or dst_addr == "0x0000") and (src_addr in device_addresses and dst_addr in device_addresses):
                timestamp = float(packet.sniff_time.timestamp())
                src_dst_pair = tuple(sorted((src_addr, dst_addr)))

                if src_dst_pair in previous_timestamps:
                    inter_arrival = timestamp - previous_timestamps[src_dst_pair]
                    inter_arrival_times[protocol][src_dst_pair].append(inter_arrival)
                previous_timestamps[src_dst_pair] = timestamp
    except AttributeError:
        continue

capture.close()

# statistics for packet lengths for each protocol
print("\nPacket Length Statistics for Each Protocol")
for protocol, lengths in packet_lengths.items():
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    median_length = np.median(lengths)
    print(f"\nProtocol: {protocol}")
    print(f"Average Packet Length: {avg_length:.2f} bytes")
    print(f"Standard Deviation of Packet Length: {std_length:.2f} bytes")
    print(f"Median Packet Length: {median_length} bytes")

# statistics for inter arrival times for device pairs
print("\nInter-Arrival Time Statistics for Each Protocol and Host Pair (Filtered by Device Addresses)")
for protocol, pairs in inter_arrival_times.items():
    for pair, times in pairs.items():
        avg_time = np.mean(times)
        std_time = np.std(times)
        median_time = np.median(times)
        print(f"\nProtocol: {protocol}, Host Pair: {pair}")
        print(f"Average Inter-Arrival Time: {avg_time:.4f} seconds")
        print(f"Standard Deviation of Inter-Arrival Time: {std_time:.4f} seconds")
        print(f"Median Inter-Arrival Time: {median_time:.4f} seconds")

# CDF for packet sizes for each protocol
for protocol, lengths in packet_lengths.items():
    sorted_lengths = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_lengths, cdf, marker='.', linestyle='none')
    plt.title(f'Cumulative Distribution Function (CDF) of Packet Sizes\nProtocol: {protocol}')
    plt.xlabel('Packet Size (bytes)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.show()

# CDF for all packet sizes combined
all_packet_sizes = []
for sizes in packet_lengths.values():
    all_packet_sizes.extend(sizes)

sorted_all_sizes = np.sort(all_packet_sizes)
cdf_all_sizes = np.arange(1, len(sorted_all_sizes) + 1) / len(sorted_all_sizes)

plt.figure(figsize=(10, 6))
plt.plot(sorted_all_sizes, cdf_all_sizes, marker='.', linestyle='none')
plt.title('Cumulative Distribution Function (CDF) of Packet Sizes for the Entire Scenario')
plt.xlabel('Packet Size (bytes)')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()


# CCDF for inter arrival times
for protocol, pairs in inter_arrival_times.items():
    for pair, times in pairs.items():
        sorted_times = np.sort(times)
        ccdf = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)

        plt.figure(figsize=(10, 6))
        plt.plot(sorted_times, ccdf, marker='.', linestyle='none')
        plt.title(f'Complementary Cumulative Distribution Function (CCDF) of Inter-Arrival Times\nProtocol: {protocol}, Host Pair: {pair}')
        plt.xlabel('Inter-Arrival Time (seconds)')
        plt.ylabel('Complementary Cumulative Probability')
        plt.grid(True)
        plt.show()
