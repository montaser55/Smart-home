import os
import csv
import math
import argparse


def set_direction(direction):
    if direction == "A to B":
        direction_int = 1
    elif direction == "B to A":
        direction_int = -1
    else:
        direction_int = 0
    return direction_int


def load_csv_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            packet_size = int(row[0])
            direction = set_direction(row[1])
            data.append((packet_size, direction))
    return data


def calculate_euclidean_distance(row1, row2):
    return math.sqrt((row1[0] - row2[0]) ** 2 + (row1[1] - row2[1]) ** 2)


def average_distance_between_files(data1, data2):
    total_distance = 0
    count = 0

    for row1 in data1:
        for row2 in data2:
            total_distance += calculate_euclidean_distance(row1, row2)
            count += 1

    average_distance = total_distance / count if count > 0 else 0
    return average_distance


def calculate_distances_between_all_files(folder_path, output_file):
    print(f"\nCalculating Euclidean distance...")
    csv_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            csv_data[filename] = load_csv_data(file_path)

    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["File1", "File2", "Average Distance"])

        files = list(csv_data.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                file1, file2 = files[i], files[j]
                avg_distance = average_distance_between_files(csv_data[file1], csv_data[file2])
                writer.writerow([file1, file2, avg_distance])

    print(f"Saved Euclidean distances in {output_file}")
    print("---------------------------------------------\n")


def count_csv_rows(folder_path):
    csv_row_counts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                row_count = sum(1 for row in reader) - 1
                csv_row_counts[file_path] = row_count
                print(f"{filename}: {row_count} rows")

    if not csv_row_counts:
        print("No CSV files with rows found.")
        return

    max_row_count = max(csv_row_counts.values())
    print(f"\nLargest CSV row count: {max_row_count}")

    for file_path, row_count in csv_row_counts.items():
        if row_count < max_row_count:
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                rows_to_add = max_row_count - row_count
                empty_row = ["0", "0 to 0"]
                for _ in range(rows_to_add):
                    writer.writerow(empty_row)
            print(f"Padded {os.path.basename(file_path)} to {max_row_count} rows")


def main():
    parser = argparse.ArgumentParser(description="Process CSV files to calculate Euclidean distances and pad rows.")
    parser.add_argument("--scenario", type=str, required=True, help="Scenario folder name.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path to the main folder.")

    args = parser.parse_args()

    folder_path = os.path.join(args.base_path, args.scenario)
    for filename in os.listdir(folder_path):
        if not filename.endswith('.csv') and not filename.endswith('.png'):
            sub_folder_path = os.path.join(folder_path, filename)
            count_csv_rows(sub_folder_path)
            output_file = os.path.join(folder_path, f"average_pairwise_distances_{filename}.csv")
            calculate_distances_between_all_files(sub_folder_path, output_file)


if __name__ == "__main__":
    main()
