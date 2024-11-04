import pyshark
import csv
import os

def process_segment(segment_file, device_mac, coordinator_mac="0x0000"):
    capture = pyshark.FileCapture(segment_file, use_json=True, include_raw=True)
    packet_data = []

    for packet in capture:
        if 'ZBEE_NWK' in packet:  # Check for ZigBee network layer
            packet_size = int(packet.length)  # Use total packet size as before

            # Check for source and destination addresses
            src_addr = getattr(packet.zbee_nwk, 'src64', None)
            dst_addr = getattr(packet.zbee_nwk, 'dst', None)
            
            # Determine direction based on source and destination addresses
            if src_addr == device_mac and dst_addr == coordinator_mac:
                direction = "A to B"
            elif src_addr == coordinator_mac and dst_addr == device_mac:
                direction = "B to A"
            else:
                continue  # Skip packets that aren’t between the device and coordinator

            packet_data.append([packet_size, direction])

    capture.close()

    # Define output CSV file name and write data
    output_file = os.path.splitext(segment_file)[0] + ".csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Packet Size", "Direction"])
        writer.writerows(packet_data)

    print(f"Processed {segment_file} and saved to {output_file}")

# Usage example: replace with specific file path and MAC addresses as needed
process_segment("ledvance_segment_1.pcapng", "f0:d1:b8:00:00:1e:0e:db")
