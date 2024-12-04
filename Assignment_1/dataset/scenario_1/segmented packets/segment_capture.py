import subprocess

def segment_file(input_file, device_name):
    result = subprocess.run(
        ["tshark", "-r", input_file, "-q", "-z", "io,stat,0"],
        capture_output=True, text=True
    )
    print(result)
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

segment_file('../device specific packets/ledvance_to_coordinator.pcapng', 'ledvance')
segment_file('../device specific packets/nedisdoorsensor_to_coordinator.pcapng', 'nedisdoorsensor')
segment_file('../device specific packets/osarm_to_coordinator.pcapng', 'osarm')
