# how to use:
# python3 task1b1c.py --scenario 1

import pyshark
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

os.makedirs("../output/plots", exist_ok=True)

def process_packet_lengths(packet, packet_lengths):

    protocol = packet.highest_layer
    if protocol == '_WS.MALFORMED':
        return

    protocol = packet.highest_layer
    packet_length = int(packet.length)
    packet_lengths[protocol].append(packet_length)


def process_inter_arrival_times(packet, device_addresses, previous_timestamps, inter_arrival_times, motion_sensor_addresses=None):
    if hasattr(packet, 'zbee_nwk'):
        src_addr = packet.zbee_nwk.src
        dst_addr = packet.zbee_nwk.dst

        # special case for scenario 3
        if motion_sensor_addresses is not None:
            if src_addr in motion_sensor_addresses:
                src_addr = "MotionSensor"
            if dst_addr in motion_sensor_addresses:
                dst_addr = "MotionSensor"

        if (src_addr == "0x0000" or dst_addr == "0x0000") and (src_addr in device_addresses and dst_addr in device_addresses):
            timestamp = float(packet.sniff_time.timestamp())
            src_dst_pair = tuple(sorted((src_addr, dst_addr)))

            if src_dst_pair in previous_timestamps:
                inter_arrival = timestamp - previous_timestamps[src_dst_pair]
                inter_arrival_times[packet.highest_layer][src_dst_pair].append(inter_arrival)
            previous_timestamps[src_dst_pair] = timestamp


def print_packet_length_statistics(packet_lengths):
    print("\nPacket Length Statistics for Each Protocol")
    for protocol, lengths in packet_lengths.items():
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)
        median_length = np.median(lengths)
        print(f"\nProtocol: {protocol}")
        print(f"Average Packet Length: {avg_length:.2f} bytes")
        print(f"Standard Deviation of Packet Length: {std_length:.2f} bytes")
        print(f"Median Packet Length: {median_length} bytes")


def print_inter_arrival_time_statistics(inter_arrival_times):
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


def plot_cdf(packet_lengths, scenario_name):
    plt.figure(figsize=(12, 8))

    all_packet_sizes = []

    for protocol, lengths in packet_lengths.items():

        if protocol == '_WS.MALFORMED':
            continue

        if len(lengths) > 0:
            sorted_lengths = np.sort(lengths)
            cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
            plt.plot(sorted_lengths, cdf, label=f'{protocol}', linewidth=1.5)
            all_packet_sizes.extend(lengths)

    if all_packet_sizes:
        sorted_all_sizes = np.sort(all_packet_sizes)
        cdf_all_sizes = np.arange(1, len(sorted_all_sizes) + 1) / len(sorted_all_sizes)
        plt.plot(sorted_all_sizes, cdf_all_sizes, linestyle='--', linewidth=1.5, label='All Protocols', color='black')

    plt.title('Cumulative Distribution Function (CDF) of Packet Sizes by Protocol')
    plt.xlabel('Packet Size (bytes)')
    plt.ylabel('Cumulative Probability')
    plt.legend(title="Protocols", loc='lower right')
    plt.grid(True)
    plt.savefig(f"../output/plots/scenario_{scenario_name}_packet_sizes_cdf.png")
    plt.show()


def plot_ccdf(inter_arrival_times, scenario_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [2, 1]}, figsize=(14, 6))

    for protocol, pairs in inter_arrival_times.items():
        for pair, times in pairs.items():
            sorted_times = np.sort(times)
            ccdf = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)

            ax1.plot(sorted_times, ccdf, linestyle='-', linewidth=1, label=f"{protocol}, {pair}")
            ax1.set_xlim(0, 10)
            ax1.set_title('Zoomed-In CCDF (0–10 seconds)')
            ax1.set_xlabel('Inter-Arrival Time (seconds)')
            ax1.set_ylabel('Complementary Cumulative Probability')
            ax1.grid(True, linestyle="--", linewidth=0.5)

            ax2.plot(sorted_times, ccdf, linestyle='-', linewidth=1, label=f"{protocol}, {pair}")
            ax2.set_xlim(11, 500)
            ax2.set_title('Compressed CCDF (11–500 seconds)')
            ax2.set_xlabel('Inter-Arrival Time (seconds)')
            ax2.grid(True, linestyle="--", linewidth=0.5)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()

    ax1.legend(loc="upper left", fontsize=8, title="Protocols and Pairs")
    plt.tight_layout()
    plt.savefig(f"../output/plots/scenario_{scenario_name}_inter_arrival_times_ccdf.png")
    plt.show()


scenarios = {
    "1": {"file_path": "../pcap/scenario1.pcapng", "device_addresses": {"0x7290", "0xcf6c", "0x8b7c", "0x0000"}},
    "2": {"file_path": "../pcap/scenario2.pcapng", "device_addresses": {"0xe7fb", "0xcf6c", "0x8b7c", "0x0000"}},
    "3": {
        "file_path": "../pcap/scenario3.pcapng",
        "device_addresses": {"0xe6c4", "0xcf6c", "0x0000", "MotionSensor"},
        "motion_sensor_addresses": {"0xd0b9", "0xdc52", "0xb547", "0xacba", "0xdd43"}
    },
    "4": {"file_path": "../pcap/scenario4.pcapng", "device_addresses": {"0xd0b9", "0xcf6c", "0x8b7c", "0x0000"}},
    "5": {"file_path": "../pcap/scenario5.pcapng", "device_addresses": {"0xdd43", "0x6ef9", "0xcf6c", "0x8b7c", "0x0000"}},
    "6": {"file_path": "../pcap/scenario6.pcapng", "device_addresses": {"0x7290", "0xe6c4", "0xcf6c", "0x8b7c", "0x0000"}}
}

parser = argparse.ArgumentParser(description="Process and analyze PCAP files for a selected scenario.")
parser.add_argument('--scenario', type=str, required=True, choices=scenarios.keys(),
                    help="Scenario number to process (e.g., 1, 2, 3, ...).")

args = parser.parse_args()
selected_scenario = args.scenario

scenario = scenarios[selected_scenario]
file_path = scenario["file_path"]
device_addresses = scenario["device_addresses"]
motion_sensor_addresses = scenario.get("motion_sensor_addresses", None)

# Processing
print(f"\nProcessing Scenario {selected_scenario}\n")

packet_lengths = defaultdict(list)
inter_arrival_times = defaultdict(lambda: defaultdict(list))
previous_timestamps = {}

capture = pyshark.FileCapture(file_path)

for packet in capture:
    try:
        process_packet_lengths(packet, packet_lengths)
        process_inter_arrival_times(packet, device_addresses, previous_timestamps, inter_arrival_times, motion_sensor_addresses)
    except AttributeError:
        continue

capture.close()

print_packet_length_statistics(packet_lengths)
print_inter_arrival_time_statistics(inter_arrival_times)
plot_cdf(packet_lengths, selected_scenario)
plot_ccdf(inter_arrival_times, selected_scenario)
