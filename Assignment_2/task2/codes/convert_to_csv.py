# import os
# import pyshark
# import csv
# from collections import defaultdict
#
# devices = {
#     "frient_door_sensor": ["0xe6c4"],
#     "motion_sensor": ["0xd0b9", "0xdc52", "0xb547", "0xacba", "0xdd43"],
#     "outdoor_motion_sensor": ["0x6ef9"],
#     "ledvance": ["0xcf6c"],
#     "nedis_door_sensor": ["0x7290"],
#     "osram": ["0x8b7c"],
#     "water_leak_sensor": ["0xe7fb"],
#     "coordinator": ["0x0000"]
# }
#
# coordinator_mac = "0x0000"
#
# def get_device_name(mac_address):
#     for device_name, mac_list in devices.items():
#         if mac_address in mac_list:
#             return device_name
#     return "Unknown Device"
#
# def get_all_device_mac_addresses():
#     mac_addresses = []
#     for mac_list in devices.values():
#         mac_addresses.extend(mac_list)
#     return mac_addresses
#
# def extract_zigbee_data(pcap_file):
#     capture = pyshark.FileCapture(pcap_file, display_filter="zbee_nwk")
#     packet_data = []
#     interarrival_data = []
#     last_packet_times = defaultdict(dict)
#
#     try:
#         for packet in capture:
#             if hasattr(packet, "ZBEE_NWK"):
#                 source_mac = packet.ZBEE_NWK.src
#                 destination_mac = packet.ZBEE_NWK.dst
#
#                 if source_mac not in get_all_device_mac_addresses() or destination_mac not in get_all_device_mac_addresses():
#                     continue
#
#                 if source_mac != coordinator_mac and destination_mac != coordinator_mac:
#                     continue
#
#                 direction = "send" if source_mac == coordinator_mac else "receive"
#                 packet_size = int(packet.length)
#                 source_device = get_device_name(source_mac)
#                 destination_device = get_device_name(destination_mac)
#
#                 packet_data.append([direction, source_device, destination_device, packet_size])
#
#                 pair_key = tuple(sorted([source_mac, destination_mac]))
#
#                 if pair_key in last_packet_times:
#                     last_time = last_packet_times[pair_key]
#                     inter_arrival_time = (packet.sniff_time - last_time).total_seconds()
#                 else:
#                     inter_arrival_time = None
#
#                 last_packet_times[pair_key] = packet.sniff_time
#
#                 interarrival_data.append([direction, source_device, destination_device, inter_arrival_time])
#     except Exception as e:
#         print(f"Error occurred while processing the capture: {e}")
#     finally:
#         capture.close()
#
#     return packet_data, interarrival_data
#
#
#
# def write_into_files(packet_data, interarrival_data, output_directory, scenario):
#     os.makedirs(f"{output_directory}/packet_size", exist_ok=True)
#     packet_size_csv = f"{output_directory}/packet_size/{scenario}_packet_size.csv"
#     with open(packet_size_csv, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Direction", "Source Device", "Destination Device", "Packet Size"])
#         writer.writerows(packet_data)
#     print(f"Packet size data has been written to {packet_size_csv}")
#
#     os.makedirs(f"{output_directory}/inter_arrival_time", exist_ok=True)
#     inter_arrival_csv = f"{output_directory}/inter_arrival_time/{scenario}_interarrival_times.csv"
#     with open(inter_arrival_csv, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["Inter-Arrival Time"])
#         for row in interarrival_data:
#             if row[-1] is not None:
#                 writer.writerow(row[-1:])
#     print(f"Inter-arrival time data has been written to {inter_arrival_csv}")
#
#
# output_directory = "../dataset/csv"
# folder_path = "../dataset/pcap/filtered_scenario_packets/"
#
# for filename in os.listdir(folder_path):
#     if filename.endswith('.pcapng'):
#         pcap_file = folder_path + filename
#         packet_data, interarrival_data = extract_zigbee_data(pcap_file)
#         scenario = filename.split(".")[0].split("_")[0]
#         write_into_files(packet_data, interarrival_data,  output_directory, scenario)
import json
import os
import pyshark
import csv
from collections import defaultdict
from datetime import timedelta

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

def extract_zigbee_data(pcap_file):
    capture = pyshark.FileCapture(pcap_file, display_filter="zbee_nwk")
    interarrival_data = defaultdict(list)  # Group inter-arrival times by interval
    packet_data = defaultdict(list)  # Group packet size and direction pairs by interval
    last_packet_times = defaultdict(dict)

    try:
        for packet in capture:
            if hasattr(packet, "ZBEE_NWK"):
                source_mac = packet.ZBEE_NWK.src
                destination_mac = packet.ZBEE_NWK.dst

                if source_mac not in get_all_device_mac_addresses() or destination_mac not in get_all_device_mac_addresses():
                    continue

                if source_mac != coordinator_mac and destination_mac != coordinator_mac:
                    continue

                # Extract packet data
                direction = "send" if source_mac == coordinator_mac else "receive"
                packet_size = int(packet.length)
                source_device = get_device_name(source_mac)
                destination_device = get_device_name(destination_mac)

                pair_key = tuple(sorted([source_mac, destination_mac]))
                current_time = packet.sniff_time

                # Compute inter-arrival time
                if pair_key in last_packet_times:
                    last_time = last_packet_times[pair_key]
                    inter_arrival_time = (current_time - last_time).total_seconds()
                else:
                    inter_arrival_time = None

                last_packet_times[pair_key] = current_time

                # Group data into 5-minute intervals
                interval_key = current_time.replace(second=0, microsecond=0) - timedelta(minutes=current_time.minute % 5)

                if inter_arrival_time is not None:
                    interarrival_data[interval_key].append(inter_arrival_time)
                # Store direction and packet size together as a tuple
                packet_data[interval_key].append([direction, packet_size])
    except Exception as e:
        print(f"Error occurred while processing the capture: {e}")
    finally:
        capture.close()

    return interarrival_data, packet_data


def write_into_files(interarrival_data, packet_data, output_directory, scenario):
    # Write inter-arrival times
    os.makedirs(f"{output_directory}/inter_arrival_time", exist_ok=True)
    inter_arrival_csv = f"{output_directory}/inter_arrival_time/{scenario}_interarrival_times.csv"
    with open(inter_arrival_csv, mode='w') as f:
        f.write("Inter-Arrival Times\n")
        for interval, times in interarrival_data.items():
            times_str = ",".join(map(str, times))
            f.write(f"[{times_str}]\n")
    print(f"Inter-arrival time data has been written to {inter_arrival_csv}")

    # Write packet size and direction as a group
    os.makedirs(f"{output_directory}/packet_size_direction", exist_ok=True)
    packet_csv = f"{output_directory}/packet_size_direction/{scenario}_packet_size_direction.csv"
    with open(packet_csv, mode='w') as f:
        f.write("Packet Size and Direction\n")
        for interval, data in packet_data.items():
            data_str = ",".join([f"[{d[0]},{d[1]}]" for d in data])  # Convert nested lists to strings
            f.write(f"[{data_str}]\n")  # Write manually, no enclosing quotes
    print(f"Packet size and direction data has been written to {packet_csv}")


output_directory = "../dataset/csv"
folder_path = "../../../Assignment_1/dataset/scenario_6/device specific packets/"

for filename in os.listdir(folder_path):
    if filename.endswith('.pcapng'):
        pcap_file = folder_path + filename
        interarrival_data, packet_data = extract_zigbee_data(pcap_file)
        scenario = filename.split(".")[0].split("_")[0]
        write_into_files(interarrival_data, packet_data, output_directory, scenario)
