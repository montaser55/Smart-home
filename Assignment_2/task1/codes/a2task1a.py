# how to use:
# python3 task1a.py

import pyshark
from collections import Counter
import argparse

def calculate_fractions(counter, total_packet):
    fractions = {}
    for item, count in counter.items():
        fraction = count / total_packet
        fractions[item] = fraction
    return fractions

def process_file(fp):
    capture = pyshark.FileCapture(fp)
    protocol_count = Counter()
    message_type_count = Counter()
    total_packet = 0

    for packet in capture:
        total_packet += 1

        if hasattr(packet, 'highest_layer'):
            protocol = packet.highest_layer
            protocol_count[protocol] += 1

        if hasattr(packet, 'zbee_nwk'):
            message_type = packet.zbee_nwk.frame_type
            message_type_count[message_type] += 1

    capture.close()
    return protocol_count, message_type_count, total_packet

def print_and_save_results(file, header, total_packets, protocol_counter, protocol_fractions, message_type_counter, message_type_fractions, delimiter):
    file.write(header + "\n")
    file.write(f"Total Packets: {total_packets}\n")
    file.write(f"Protocol Counts: {protocol_counter}\n")
    file.write(f"Protocol Fractions: {protocol_fractions}\n")
    file.write(f"Message Type Counts: {message_type_counter}\n")
    file.write(f"Message Type Fractions: {message_type_fractions}\n")
    file.write("\n" + delimiter + "\n")

    print(header)
    print("Total Packets:", total_packets)
    print("Protocol Counts:", protocol_counter)
    print("Protocol Fractions:", protocol_fractions)
    print("Message Type Counts:", message_type_counter)
    print("Message Type Fractions:", message_type_fractions)
    print("\n" + delimiter + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze PCAP files and calculate statistics.")
    parser.add_argument('--base_directory', type=str, required=True, help="Base directory containing the PCAP files.")
    parser.add_argument('--file_names', nargs='+', default=['scenario1.pcapng', 'scenario2.pcapng', 'scenario3.pcapng', 'scenario4.pcapng', 'scenario5.pcapng', 'scenario6.pcapng'], help="List of PCAP file names to analyze.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output file.")
    parser.add_argument('--delimiter', type=str, default="=" * 50, help="Delimiter for separating sections in the output.")
    args = parser.parse_args()

    base_directory = args.base_directory
    file_names = args.file_names
    output_file_path = args.output_file
    delimiter = args.delimiter

    file_paths = [f"{base_directory}/{file_name}" for file_name in file_names]

    cumulative_protocol_counter = Counter()
    cumulative_message_type_counter = Counter()
    cumulative_total_packets = 0

    with open(output_file_path, "w") as file:
        for file_path in file_paths:
            try:
                protocol_counter, message_type_counter, total_packets = process_file(file_path)

                protocol_fractions = calculate_fractions(protocol_counter, total_packets)
                message_type_fractions = calculate_fractions(message_type_counter, total_packets)

                print_and_save_results(file, f"Results for {file_path}", total_packets, protocol_counter, protocol_fractions, message_type_counter, message_type_fractions, delimiter)

                cumulative_protocol_counter.update(protocol_counter)
                cumulative_message_type_counter.update(message_type_counter)
                cumulative_total_packets += total_packets

            except FileNotFoundError:
                print(f"File not found: {file_path}")
                file.write(f"File not found: {file_path}\n")
                file.write("\n" + delimiter + "\n")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                file.write(f"Error processing {file_path}: {e}\n")
                file.write("\n" + delimiter + "\n")

        cumulative_protocol_fractions = calculate_fractions(cumulative_protocol_counter, cumulative_total_packets)
        cumulative_message_type_fractions = calculate_fractions(cumulative_message_type_counter, cumulative_total_packets)

        print_and_save_results(file, "Cumulative Results Across All Scenarios", cumulative_total_packets, cumulative_protocol_counter, cumulative_protocol_fractions, cumulative_message_type_counter, cumulative_message_type_fractions, delimiter)


if __name__ == "__main__":
    main()