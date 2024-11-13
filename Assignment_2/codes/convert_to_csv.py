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
    data_by_pair = defaultdict(list)
    interarrival_times_by_pair = defaultdict(list)  # Dictionary to store inter-arrival times

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
                pair_key = tuple(sorted([source_mac, destination_mac]))

                source_device = get_device_name(pair_key[0])
                destination_device = get_device_name(pair_key[1])
                pair_key_string = f"{source_device},{destination_device}"
                data_by_pair[pair_key_string].append([direction, source_mac, destination_mac, packet_size])


                if interarrival_times_by_pair[pair_key_string]:
                    last_time = interarrival_times_by_pair[pair_key_string][-1][-1]
                    inter_arrival_time = (packet.sniff_time - last_time).total_seconds()
                else:
                    inter_arrival_time = None

                interarrival_times_by_pair[pair_key_string].append(
                    [direction, source_mac, destination_mac, packet_size, inter_arrival_time, packet.sniff_time]
                )

    except Exception as e:
        print(f"Error occurred while processing the capture: {e}")
    finally:
        capture.close()

    for pair_key_string, data in data_by_pair.items():
        destination_device = pair_key_string.split(",")[1]
        output_csv = f"{output_directory}/packet_size/{scenario}_{destination_device}_packet_size.csv"
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Direction", "Source", "Destination", "Packet Size"])
            writer.writerows(data)
        print(f"Data for pair {pair_key_string} has been written to {output_csv}")

    for pair_key_string, inter_data in interarrival_times_by_pair.items():
        destination_device = pair_key_string.split(",")[1]
        inter_output_csv = f"{output_directory}/inter_arrival_time/{scenario}_{destination_device}_interarrival_times.csv"
        flag = 0
        with open(inter_output_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Inter-Arrival Time"])
            for row in inter_data:
                if flag == 0:
                    flag = 1
                    continue
                writer.writerow(row[4:5])
        print(f"Inter-arrival time data for pair {pair_key_string} has been written to {inter_output_csv}")


output_directory = "../dataset/csv"
folder_path = "../dataset/pcap/device_specific_packets/"
for filename in os.listdir(folder_path):
    if filename.endswith('.pcapng'):
        pcap_file = folder_path + filename
        extract_zigbee_data(pcap_file, output_directory, filename.split(".")[0].split("_")[0])
