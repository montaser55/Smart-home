import pyshark
from collections import Counter

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

def  print_and_save_results(file, header, total_packets, protocol_counter, protocol_fractions, message_type_counter, message_type_fractions):
    file.write(header + "\n")
    file.write(f"Total Packets: {total_packets}\n")
    file.write(f"Protocol Counts: {protocol_counter}\n")
    file.write(f"Protocol Fractions: {protocol_fractions}\n")
    file.write(f"Message Type Counts: {message_type_counter}\n")
    file.write(f"Message Type Fractions: {message_type_fractions}\n")
    file.write("\n" + "=" * 50 + "\n")

    print(header)
    print("Total Packets:", total_packets)
    print("Protocol Counts:", protocol_counter)
    print("Protocol Fractions:", protocol_fractions)
    print("Message Type Counts:", message_type_counter)
    print("Message Type Fractions:", message_type_fractions)
    print("\n" + "=" * 50 + "\n")

base_directory = "../pcap"
file_names = ['scenario1.pcapng',
              'scenario2.pcapng',
              'scenario3.pcapng',
              'scenario4.pcapng',
              'scenario5.pcapng',
              'scenario6.pcapng']
file_paths = [f"{base_directory}/{file_name}" for file_name in file_names]

cumulative_protocol_counter = Counter()
cumulative_message_type_counter = Counter()
cumulative_total_packets = 0

with open("../output/task_1a_analysis_results.txt", "w") as file:
    for file_path in file_paths:

        protocol_counter, message_type_counter, total_packets = process_file(file_path)

        protocol_fractions = calculate_fractions(protocol_counter, total_packets)
        message_type_fractions = calculate_fractions(message_type_counter, total_packets)

        print_and_save_results(file, f"Results for {file_path}", total_packets, protocol_counter, protocol_fractions, message_type_counter, message_type_fractions)

        cumulative_protocol_counter.update(protocol_counter)
        cumulative_message_type_counter.update(message_type_counter)
        cumulative_total_packets += total_packets

    cumulative_protocol_fractions = calculate_fractions(cumulative_protocol_counter, cumulative_total_packets)
    cumulative_message_type_fractions = calculate_fractions(cumulative_message_type_counter, cumulative_total_packets)

    print_and_save_results(file, "Cumulative Results Across All Scenarios", cumulative_total_packets, cumulative_protocol_counter, cumulative_protocol_fractions, cumulative_message_type_counter, cumulative_message_type_fractions)
