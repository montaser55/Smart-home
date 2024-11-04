import subprocess
from datetime import timedelta

# Define segmentation function using relative time filtering
def segment_file(input_file, device_name):
    # Run tshark to find the duration of the capture
    result = subprocess.run(
        ["tshark", "-r", input_file, "-q", "-z", "io,stat,0"],
        capture_output=True, text=True
    )
    capture_duration = float(result.stdout.split("Duration: ")[1].split(" ")[0])

    # Duration of each segment (5 minutes in seconds)
    segment_duration = 5 * 60
    segment_number = 1
    start_time = 0

    while start_time < capture_duration:
        end_time = start_time + segment_duration

        # Define output filename
        segment_filename = f"{device_name}_segment_{segment_number}.pcapng"

        # Run tshark to filter packets by relative time range
        command = [
            "tshark", "-r", input_file,
            "-Y", f"frame.time_relative >= {start_time} && frame.time_relative < {end_time}",
            "-w", segment_filename
        ]
        subprocess.run(command)

        # Stop if no packets are found in this segment
        if not subprocess.run(["tshark", "-r", segment_filename, "-c", "1"]).returncode == 0:
            break

        # Update time for the next segment
        start_time = end_time
        segment_number += 1

# Run segmentation for each device file
segment_file('//Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/dataset/scenario_1/device specific packets/ledvance_to_coordinator.pcapng', 'ledvance')
segment_file('/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/dataset/scenario_1/device specific packets/doorsensor_to_coordinator.pcapng', 'doorsensor')
segment_file('/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/dataset/scenario_1/device specific packets/osarm_to_coordinator.pcapng', 'osarm')
