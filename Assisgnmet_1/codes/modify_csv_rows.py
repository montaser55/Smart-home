import os
import csv

def count_csv_rows(folder_path):
    csv_row_counts = {}  # Dictionary to store row counts for each file

    # Step 1: Calculate row count for each CSV file and delete empty ones
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                row_count = sum(1 for row in reader) - 1  # Exclude header
            # If CSV has 0 rows, delete it
            if row_count <= 0:
                os.remove(file_path)
                print(f"Deleted {filename} (0 rows)")
            else:
                csv_row_counts[file_path] = row_count
                print(f"{filename}: {row_count} rows")

    # Step 2: Find the maximum row count among remaining files
    if not csv_row_counts:
        print("No CSV files with rows found.")
        return

    max_row_count = max(csv_row_counts.values())
    print(f"\nLargest CSV row count: {max_row_count}")

    # Step 3: Pad other CSV files to match the maximum row count
    for file_path, row_count in csv_row_counts.items():
        if row_count < max_row_count:
            with open(file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                # Calculate number of rows to add
                rows_to_add = max_row_count - row_count
                empty_row = ["0", "0 to 0"]  # Adjust number of empty cells as per column count
                for _ in range(rows_to_add):
                    writer.writerow(empty_row)
            print(f"Padded {os.path.basename(file_path)} to {max_row_count} rows")

# Example usage
folder_path = '/Users/montasermajid/Documents/Btu Cottbus/Smart Home/segmented into csv'  # Replace with your actual folder path
count_csv_rows(folder_path)
