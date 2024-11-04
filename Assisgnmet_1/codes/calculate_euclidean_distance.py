import os
import csv
import math

def load_csv_data(file_path):
    """Load CSV data and map direction values to numerical representations."""
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            packet_size = int(row[0])
            direction = 1 if row[1] == "A to B" else -1
            data.append((packet_size, direction))
    return data

def calculate_euclidean_distance(row1, row2):
    """Calculate Euclidean distance between two rows."""
    return math.sqrt((row1[0] - row2[0]) ** 2 + (row1[1] - row2[1]) ** 2)

def average_distance_between_files(data1, data2):
    """Calculate the average Euclidean distance between every row in two datasets."""
    total_distance = 0
    count = 0

    for row1 in data1:
        for row2 in data2:
            total_distance += calculate_euclidean_distance(row1, row2)
            count += 1

    average_distance = total_distance / count if count > 0 else 0
    return average_distance

def calculate_distances_between_all_files(folder_path, output_file):
    """Calculate average Euclidean distance between each pair of CSV files."""
    # Step 1: Load data from each CSV file
    csv_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            csv_data[filename] = load_csv_data(file_path)

    # Step 2: Prepare output CSV for distances
    with open(output_file, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(["File1", "File2", "Average Distance"])  # Header

        # Step 3: Calculate average distances between each pair of files
        files = list(csv_data.keys())
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                file1, file2 = files[i], files[j]
                avg_distance = average_distance_between_files(csv_data[file1], csv_data[file2])
                writer.writerow([file1, file2, avg_distance])

# Example usage
folder_path = '/Users/montasermajid/Documents/Btu Cottbus/Smart Home/segmented into csv'  # Replace with your actual folder path
output_file = 'average_pairwise_distances.csv'
calculate_distances_between_all_files(folder_path, output_file)
