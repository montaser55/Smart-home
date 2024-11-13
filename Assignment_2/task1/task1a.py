import pyshark
from collections import Counter

file_paths = ['scenario1.pcapng', 'scenario2.pcapng', 'scenario3.pcapng', 'scenario4.pcapng', 'scenario5.pcapng', 'scenario6.pcapng']

cumulative_protocol_counter = Counter()
cumulative_message_type_counter = Counter()
cumulative_total_packets = 0

with open("analysis_results.txt", "w") as file:
    for file_path in file_paths:
        print(f"Processing file: {file_path}")

        capture = pyshark.FileCapture(file_path)

        protocol_counter = Counter()
        message_type_counter = Counter()
        total_packets = 0

        for packet in capture:
            total_packets += 1

            if hasattr(packet, 'highest_layer'):
                protocol = packet.highest_layer
                protocol_counter[protocol] += 1

            if hasattr(packet, 'zbee_nwk'):
                message_type = packet.zbee_nwk.frame_type
                message_type_counter[message_type] += 1

        # Calculate fractions for this specific scenario
        protocol_fractions = {protocol: count / total_packets for protocol, count in protocol_counter.items()}
        message_type_fractions = {msg_type: count / total_packets for msg_type, count in message_type_counter.items()}

        print("Total Packets:", total_packets)
        print("Protocol Counts:", protocol_counter)
        print("Protocol Fractions:", protocol_fractions)
        print("Message Type Counts:", message_type_counter)
        print("Message Type Fractions:", message_type_fractions)
        print("\n" + "=" * 40 + "\n")

        # Accumulate the results into the cumulative counters
        cumulative_protocol_counter.update(protocol_counter)
        cumulative_message_type_counter.update(message_type_counter)
        cumulative_total_packets += total_packets

        capture.close()

    # Calculate cumulative fractions across all scenarios
    cumulative_protocol_fractions = {protocol: count / cumulative_total_packets for protocol, count in cumulative_protocol_counter.items()}
    cumulative_message_type_fractions = {msg_type: count / cumulative_total_packets for msg_type, count in cumulative_message_type_counter.items()}

    file.write("Cumulative Results Across All Scenarios\n")
    file.write(f"Total Packets: {cumulative_total_packets}\n")
    file.write(f"Cumulative Protocol Counts: {cumulative_protocol_counter}\n")
    file.write(f"Cumulative Protocol Fractions: {cumulative_protocol_fractions}\n")
    file.write(f"Cumulative Message Type Counts: {cumulative_message_type_counter}\n")
    file.write(f"Cumulative Message Type Fractions: {cumulative_message_type_fractions}\n")

    print("Cumulative Results Across All Scenarios")
    print("Total Packets:", cumulative_total_packets)
    print("Cumulative Protocol Counts:", cumulative_protocol_counter)
    print("Cumulative Protocol Fractions:", cumulative_protocol_fractions)
    print("Cumulative Message Type Counts:", cumulative_message_type_counter)
    print("Cumulative Message Type Fractions:", cumulative_message_type_fractions)
