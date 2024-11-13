import numpy as np
import random
import argparse


def encode_direction(direction):
    return 0 if direction == 'receive' else 1


def decode_direction(encoded_value):
    return 'receive' if encoded_value == 0 else 'send'


def min_max_normalization(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data, min_vals, max_vals


def min_max_denormalization(data, min_vals, max_vals):
    return data * (max_vals - min_vals) + min_vals


def z_score_normalization(data):
    mean_vals = np.mean(data, axis=0)
    std_vals = np.std(data, axis=0)
    normalized_data = (data - mean_vals) / std_vals
    return normalized_data, mean_vals, std_vals


def z_score_denormalization(data, mean_vals, std_vals):
    return data * std_vals + mean_vals


def manhattan_distance(sample1, sample2):
    return np.sum(np.abs(sample1 - sample2))


def find_k_nearest_neighbors(data, sample, k):
    distances = [(i, manhattan_distance(sample, other_sample)) for i, other_sample in enumerate(data) if
                 not np.array_equal(sample, other_sample)]
    distances.sort(key=lambda x: x[1])
    return [data[i] for i, _ in distances[:k]]


def generate_synthetic_samples(data, k, total_synthetic_samples):
    synthetic_data = []
    for i, sample in enumerate(data):
        if i < total_synthetic_samples:
            k_neighbors = find_k_nearest_neighbors(data, sample, k)
            neighbor = random.choice(k_neighbors)
            diff = neighbor - sample
            random_scale = random.uniform(0, 1)
            synthetic_sample = sample + random_scale * diff
            synthetic_data.append(synthetic_sample)
    return np.array(synthetic_data)


def load_data(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip().split(',')

    if "Inter-Arrival Time" in header:
        raw_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        input_data = raw_data.reshape(-1, 1)
        return input_data, "inter_arrival"
    elif "Direction" in header:
        raw_data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=str)
        directions = np.array([encode_direction(row[0]) for row in raw_data])  # Encode Direction column
        packet_sizes = np.array([float(row[3]) for row in raw_data])  # Use Packet Size as float
        input_data = np.column_stack((directions, packet_sizes))  # Combine into 2D array
        return input_data, "direction_packet_size"
    else:
        raise ValueError("Unknown CSV format.")


def main(input_data, data_type, normalization_method, k_values, synthetic_sample_percentage):
    # Normalize data and capture necessary values for denormalization
    if normalization_method == 'min_max':
        data, min_vals, max_vals = min_max_normalization(input_data)
        param1, param2 = min_vals, max_vals
        denormalize_fn = min_max_denormalization
    elif normalization_method == 'z_score':
        data, mean_vals, std_vals = z_score_normalization(input_data)
        param1, param2 = mean_vals, std_vals
        denormalize_fn = z_score_denormalization
    else:
        raise ValueError("Invalid normalization method specified.")

    # Generate synthetic samples
    total_synthetic_samples = int(synthetic_sample_percentage * len(data) / 100)
    synthetic_datasets = {}
    for k in k_values:
        synthetic_data = generate_synthetic_samples(data, k, total_synthetic_samples)
        # Denormalize synthetic data before saving
        synthetic_datasets[f"synthetic_k_{k}"] = denormalize_fn(synthetic_data, param1, param2)

    return synthetic_datasets, data_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Zigbee Network Data")
    parser.add_argument("--data_file", type=str, required=True, help="Path to input dataset (CSV format)")
    parser.add_argument("--normalization", type=str, choices=['min_max', 'z_score'], required=True,
                        help="Normalization method")
    parser.add_argument("--k_values", nargs='+', type=int, default=[3, 5],
                        help="List of k values for nearest neighbors")
    parser.add_argument("--synthetic_percentage", type=int, default=100,
                        help="Percentage of synthetic samples relative to the real dataset size")

    args = parser.parse_args()

    # Load dataset and determine data type
    input_data, data_type = load_data(args.data_file)
    synthetic_datasets, data_type = main(input_data, data_type, args.normalization, args.k_values,
                                         args.synthetic_percentage)

    # Save synthetic datasets
    for key, synthetic_data in synthetic_datasets.items():
        if data_type == "inter_arrival":
            np.savetxt(f"{key}.csv", synthetic_data, delimiter=',', fmt='%f', header="Inter-Arrival Time",
                       comments="")
        elif data_type == "direction_packet_size":
            decoded_directions = [decode_direction(int(round(val[0]))) for val in synthetic_data]
            output_data = np.column_stack((decoded_directions, synthetic_data[:, 1]))
            np.savetxt(f"{key}.csv", output_data, delimiter=',', fmt='%s', header="Direction,Packet Size",
                       comments="")
        print(f"Saved {key}.csv")
