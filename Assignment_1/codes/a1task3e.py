import csv
import os
import argparse
import matplotlib.pyplot as plt


def load_distances(file_path):
    distances = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            distances.append(float(row[2]))
    return distances


def compute_histogram(data, bin_count):
    min_val = min(data)
    max_val = max(data)
    bin_width = (max_val - min_val) / bin_count

    bins = [min_val + i * bin_width for i in range(bin_count + 1)]
    counts = [0] * bin_count

    for value in data:
        for i in range(bin_count):
            if bins[i] <= value < bins[i + 1]:
                counts[i] += 1
                break
        if value == max_val:
            counts[-1] += 1

    return bins, counts


def plot_histogram(bins, counts, save_path):
    plt.figure()
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]

    plt.bar(bin_centers, counts, width=(bins[1] - bins[0]), edgecolor="black", align="center")
    plt.xlabel("Euclidean Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Euclidean Distances")

    plt.xticks(range(0, int(bins[-1]) + 5, 5))
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Histogram saved as {save_path}")
    plt.close()


def process_scenario(scenario_folder, bin_count):
    for filename in os.listdir(scenario_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(scenario_folder, filename)
            distances = load_distances(file_path)
            bins, counts = compute_histogram(distances, bin_count)
            save_path = os.path.splitext(file_path)[0] + "_histogram.png"
            plot_histogram(bins, counts, save_path)


def main():
    parser = argparse.ArgumentParser(description="Generate histograms for Euclidean distances in CSV files.")
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Base path to the folder containing scenario data."
    )

    args = parser.parse_args()

    for index in range(1,7):
        scenario = f'scenario_{index}'
        scenario_folder = os.path.join(args.base_path, scenario)
        if os.path.exists(scenario_folder):
            print(f"Processing {scenario}...")
            process_scenario(scenario_folder, 10)
        else:
            print(f"Scenario folder {scenario_folder} does not exist.")


if __name__ == "__main__":
    main()
