import csv
import os

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

    plt.xticks(range(0, int(bins[-1])+5, 5))
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Histogram saved as {save_path}")

for index in range(1,7):
    scenario = f'scenario_{index}'
    folder_path = "/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/output/" + scenario
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = f'/Users/montasermajid/Documents/Btu Cottbus/Smart-home/Assisgnmet_1/output/{scenario}/{filename}'
            distances = load_distances(file_path)
            bin_count = 10
            bins, counts = compute_histogram(distances, bin_count)
            save_path = file_path.split('.')[0] + "_histogram.png"
            plot_histogram(bins, counts, save_path)
