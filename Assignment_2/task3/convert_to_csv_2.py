import pyshark
import csv

def extract_zigbee_data(pcap_file, output_csv):
    capture = pyshark.FileCapture(pcap_file, display_filter="zbee_nwk")

    data = []
    for packet in capture:

        source_mac = packet.ZBEE_NWK.src
        destination_mac = packet.ZBEE_NWK.dst

        if source_mac < destination_mac:
            direction = "incoming"
        else:
            direction = "outgoing"

        packet_size = int(packet.length)
        data.append([direction, packet_size])

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Direction", "PacketSize"])
        writer.writerows(data)

    print(f"Data has been successfully written to {output_csv}")


pcap_file = 'ledvance_to_coordinator.pcapng'
output_csv = 'output_file.csv'
extract_zigbee_data(pcap_file, output_csv)
