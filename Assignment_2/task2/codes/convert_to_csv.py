import os
import pyshark
import csv
from collections import defaultdict

devices = {
    "frient_door_sensor": ["0xe6c4"],
    "motion_sensor": ["0xd0b9", "0xdc52", "0xb547", "0xacba", "0xdd43"],
    "outdoor_motion_sensor": ["0x6ef9"],
    "ledvance": ["0xcf6c"],
    "nedis_door_sensor": ["0x7290"],
    "osram": ["0x8b7c"],
    "water_leak_sensor": ["0xe7fb"],
    "coordinator": ["0x0000"]
}

coordinator_mac = "0x0000"

def get_device_name(mac_address):
    for device_name, mac_list in devices.items():
        if mac_address in mac_list:
            return device_name
    return "Unknown Device"

def get_all_device_mac_addresses():
    mac_addresses = []
    for mac_list in devices.values():
        mac_addresses.extend(mac_list)
    return mac_addresses

def extract_zigbee_data(pcap_file, output_directory, scenario):
    capture = pyshark.FileCapture(pcap_file, display_filter="zbee_nwk")
    packet_data = []  # Consolidated packet size data
    interarrival_data = []  # Consolidated inter-arrival time data
    last_packet_times = defaultdict(dict)  # Track last packet time for each device pair

    try:
        for packet in capture:
            if hasattr(packet, "ZBEE_NWK"):
                source_mac = packet.ZBEE_NWK.src
                destination_mac = packet.ZBEE_NWK.dst

                if source_mac not in get_all_device_mac_addresses() or destination_mac not in get_all_device_mac_addresses():
                    continue

                if source_mac != coordinator_mac and destination_mac != coordinator_mac:
                    continue

                direction = "send" if source_mac == coordinator_mac else "receive"
                packet_size = int(packet.length)
                source_device = get_device_name(source_mac)
                destination_device = get_device_name(destination_mac)

                # Add to packet data
                packet_data.append([direction, source_device, destination_device, packet_size])

                # Calculate inter-arrival time

                pair_key = tuple(sorted([source_mac, destination_mac]))

                if pair_key in last_packet_times:
                    last_time = last_packet_times[pair_key]
                    inter_arrival_time = (packet.sniff_time - last_time).total_seconds()
                else:
                    inter_arrival_time = None

                last_packet_times[pair_key] = packet.sniff_time

                # Add to inter-arrival time data
                interarrival_data.append([direction, source_device, destination_device, inter_arrival_time])
    except Exception as e:
        print(f"Error occurred while processing the capture: {e}")
    finally:
        capture.close()

    # Write consolidated packet size data
    os.makedirs(f"{output_directory}/packet_size", exist_ok=True)
    packet_size_csv = f"{output_directory}/packet_size/{scenario}_packet_size.csv"
    with open(packet_size_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Direction", "Source Device", "Destination Device", "Packet Size"])
        writer.writerows(packet_data)
    print(f"Packet size data has been written to {packet_size_csv}")

    # Write consolidated inter-arrival time data
    os.makedirs(f"{output_directory}/inter_arrival_time", exist_ok=True)
    inter_arrival_csv = f"{output_directory}/inter_arrival_time/{scenario}_interarrival_times.csv"
    with open(inter_arrival_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Inter-Arrival Time"])
        for row in interarrival_data:
            if row[-1] is not None:  # Skip None values for inter-arrival times
                writer.writerow(row[-1:])
    print(f"Inter-arrival time data has been written to {inter_arrival_csv}")

output_directory = "../dataset/csv3"
folder_path = "../dataset/pcap/filtered_scenario_packets/"
for filename in os.listdir(folder_path):
    if filename.endswith('.pcapng'):
        pcap_file = folder_path + filename
        extract_zigbee_data(pcap_file, output_directory, filename.split(".")[0].split("_")[0])
