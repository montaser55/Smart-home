import pyshark
import csv


# Function to extract relevant information from Zigbee packets
def extract_zigbee_data(pcap_file, output_csv):
    # Open the pcap file using pyshark
    capture = pyshark.FileCapture(pcap_file, display_filter="zbee_nwk")

    # List to store the results
    data = []
    # Iterate over all packets
    for packet in capture:
        # Check if the packet has Zigbee data and extract necessary information
        print(packet.ZBEE_NWK)
        # Extract the source and destination MAC addresses
        source_mac = packet.ZBEE_NWK.src
        destination_mac = packet.ZBEE_NWK.dst

        # Determine the direction (send or receive)
        direction = "incoming" if source_mac < destination_mac else "outgoing"

        # Extract the size of the Zigbee packet (total length including headers)
        packet_size = int(packet.length)

        # Append the extracted data to the list
        data.append([direction, packet_size])

    # Write the extracted data into a CSV file
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Direction", "PacketSize"])
        writer.writerows(data)

    print(f"Data has been successfully written to {output_csv}")


# Example usage
pcap_file = 'ledvance_to_coordinator.pcapng'
output_csv = 'output_file.csv'
extract_zigbee_data(pcap_file, output_csv)