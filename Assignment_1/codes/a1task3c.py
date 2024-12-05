# how to use:
# python3 make_into_csv.py --scenario 6

import pyshark
import csv
import os
import argparse

DEVICE_MACS = {
    "frientdoorsensor": ["0xe6c4"],
    "motionsensor": ["0xd0b9", "0xdc52", "0xb547", "0xacba", "0xdd43"],
    "outdoormotionsensor": ["0x6ef9"],
    "ledvance": ["0xcf6c"],
    "nedisdoorsensor": ["0x7290"],
    "osarm": ["0x8b7c"],
    "waterleaksensor": ["0xe7fb"]
}

def process_segment(segment_file, device_name, mac_addresses, directory_name, scenario_number, output_directory, coordinator_mac="0x0000"):
    capture = pyshark.FileCapture(segment_file, use_json=True, include_raw=True)
    packet_data = []

    for packet in capture:
        if 'ZBEE_NWK' in packet:
            packet_size = int(packet.length)

            src_addr = getattr(packet.zbee_nwk, 'src', None)
            dst_addr = getattr(packet.zbee_nwk, 'dst', None)

            if src_addr in mac_addresses and dst_addr == coordinator_mac:
                direction = "A to B"
            elif src_addr == coordinator_mac and dst_addr in mac_addresses:
                direction = "B to A"
            else:
                continue

            packet_data.append([packet_size, direction])

    capture.close()

    scenario_path = f'scenario_{scenario_number}/{directory_name}/'
    full_output_directory = os.path.join(output_directory, scenario_path)
    output_file = os.path.join(full_output_directory, f"{os.path.splitext(os.path.basename(segment_file))[0]}.csv")

    os.makedirs(full_output_directory, exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Packet Size", "Direction"])
        writer.writerows(packet_data)

    print(f"Processed {segment_file} and saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Zigbee network traffic data.")
    parser.add_argument('--base_directory', type=str, required=True, help="Base directory.")
    parser.add_argument('--scenario', required=True, type=int, help="Scenario number (e.g., 1, 2, etc.)")
    parser.add_argument('--start', type=int, default=1, help="Start index of the pcapng files (default: 1).")
    parser.add_argument('--end', type=int, default=48, help="End index of the pcapng files (default: 48).")
    parser.add_argument('--coordinator_mac', default="0x0000", help="Coordinator MAC address (default: 0x0000).")

    args = parser.parse_args()

    input_directory = f"{args.base_directory}/dataset/scenario_{args.scenario}/segmented packets"
    output_directory = f"{args.base_directory}/output/"

    for device_name, mac_addresses in DEVICE_MACS.items():
        for index in range(args.start, args.end + 1):
            segment_file = os.path.join(input_directory, f'{device_name}_segment_{index}.pcapng')
            if os.path.exists(segment_file):
                process_segment(segment_file, device_name, mac_addresses, device_name, args.scenario, output_directory, args.coordinator_mac)
            else:
                print(f"File not found: {segment_file}")


if __name__ == "__main__":
    main()