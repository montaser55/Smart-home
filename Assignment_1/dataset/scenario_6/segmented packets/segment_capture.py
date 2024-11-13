import subprocess
from datetime import timedelta


def segment_file(input_file, device_name):
    result = subprocess.run(
        ["tshark", "-r", input_file, "-q", "-z", "io,stat,0"],
        capture_output=True, text=True
    )
    capture_duration = float(result.stdout.split("Duration: ")[1].split(" ")[0])
    segment_duration = 5 * 60
    segment_number = 1
    start_time = 0

    while start_time < capture_duration:
        end_time = start_time + segment_duration
        segment_filename = f"{device_name}_segment_{segment_number}.pcapng"
        command = [
            "tshark", "-r", input_file,
            "-Y", f"frame.time_relative >= {start_time} && frame.time_relative < {end_time}",
            "-w", segment_filename
        ]
        subprocess.run(command)
        start_time = end_time
        segment_number += 1

segment_file('/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/dataset/scenario_6/device specific packets/ledvance_to_coordinator.pcapng', 'ledvance')
segment_file('/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/dataset/scenario_6/device specific packets/nedisdoorsensor_to_coordinator.pcapng', 'nedisdoorsensor')
segment_file('/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/dataset/scenario_6/device specific packets/scenario1_osarm_to_coordinator.pcapng', 'osarm')
segment_file('/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/dataset/scenario_6/device specific packets/frientdoorsensor_to_coordinator.pcapng', 'frientdoorsensor')
