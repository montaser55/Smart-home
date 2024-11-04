import pyshark
import csv
import os

def process_segment(segment_file, device_mac, directory_name, coordinator_mac="0x0000"):
    capture = pyshark.FileCapture(segment_file, use_json=True, include_raw=True)
    packet_data = []

    for packet in capture:
        if 'ZBEE_NWK' in packet:
            packet_size = int(packet.length)

            src_addr = getattr(packet.zbee_nwk, 'src', None)
            dst_addr = getattr(packet.zbee_nwk, 'dst', None)

            if src_addr in device_mac and dst_addr == coordinator_mac:
                direction = "A to B"
            elif src_addr == coordinator_mac and dst_addr in device_mac:
                direction = "B to A"
            else:
                continue

            packet_data.append([packet_size, direction])

    capture.close()
    #print(os.path.splitext(segment_file)[0].split('/')[-1])
    location = "../../../output/"
    scenario = f'scenario_3/{directory_name}/'
    output_directory = location + scenario
    output_file = output_directory + os.path.splitext(segment_file)[0].split('/')[-1] + ".csv"

    os.makedirs(output_directory, exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Packet Size", "Direction"])
        writer.writerows(packet_data)

    print(f"Processed {segment_file} and saved to {output_file}")

for index in range(1, 49):
    process_segment(f'../segmented packets/ledvance_segment_{index}.pcapng', ["0xcf6c"], "ledvance")
    process_segment(f'../segmented packets/doorsensor_segment_{index}.pcapng', ["0xe6c4"], "doorsensor")
    process_segment(f'../segmented packets/motionsensor_segment_{index}.pcapng', ["0xd0b9","0xdc52","0xb547","0xacba","0xdd43"],"motionsensor")

