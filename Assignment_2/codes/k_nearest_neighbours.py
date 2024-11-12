import numpy as np
import random
import argparse


# Encode direction for categorical handling
def encode_direction(direction):
    return 0 if direction == 'receive' else 1


def decode_direction(encoded_value):
    return 'receive' if encoded_value == 0 else 'send'


# Normalization Functions
def min_max_normalization(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def z_score_normalization(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)


# Manhattan Distance Calculation
def manhattan_distance(sample1, sample2):
    return np.sum(np.abs(sample1 - sample2))


# Function to find k nearest neighbors
def find_k_nearest_neighbors(data, sample, k):
    distances = []
    for i, other_sample in enumerate(data):
        if not np.array_equal(sample, other_sample):
            distances.append((i, manhattan_distance(sample, other_sample)))
    distances.sort(key=lambda x: x[1])  # Sort by distance
    return [data[i] for i, _ in distances[:k]]


# Synthetic Sample Generation
def generate_synthetic_samples(data, k, total_synthetic_samples):
    synthetic_data = []
    samples_to_generate = total_synthetic_samples // len(data)  # Evenly distribute synthetic samples across the dataset
    remaining_samples = total_synthetic_samples % len(data)  # Handle any remaining samples
    print(total_synthetic_samples, samples_to_generate)
    for i, sample in enumerate(data):
         if i < remaining_samples:
            neighbor = random.choice(find_k_nearest_neighbors(data, sample, k))
            diff = neighbor - sample
            random_scale = random.uniform(0, 1)
            synthetic_sample = sample + random_scale * diff
            synthetic_data.append(synthetic_sample)

    return np.array(synthetic_data)


# Main Function
def main(input_data, normalization_method, k_values, synthetic_sample_percentage):
    # Normalize data based on user-selected method
    if normalization_method == 'min_max':
        data = min_max_normalization(input_data)
    elif normalization_method == 'z_score':
        data = z_score_normalization(input_data)
    else:
        raise ValueError("Invalid normalization method specified.")

    # Calculate total synthetic samples to generate for the whole dataset
    total_synthetic_samples = int(synthetic_sample_percentage * len(data) / 100)
    print(total_synthetic_samples)
    synthetic_datasets = {}

    # Generate synthetic data for each value of k
    for k in k_values:
        synthetic_data = generate_synthetic_samples(data, k, total_synthetic_samples)
        synthetic_datasets[f"synthetic_k_{k}"] = synthetic_data

    return synthetic_datasets


# Command-line Interface
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

    # Load dataset
    raw_data = np.genfromtxt(args.data_file, delimiter=',', skip_header=1, dtype=None, encoding=None)

    # Convert data into numeric form
    directions = np.array([encode_direction(row[0]) for row in raw_data])  # Encode Direction column
    packet_sizes = np.array([row[3] for row in raw_data], dtype=float)  # Use Packet Size

    # Combine direction and packet size into a single array for processing
    input_data = np.column_stack((directions, packet_sizes))
    input_data = input_data[:100,:]
    print(input_data)
    # Generate synthetic datasets
    synthetic_datasets = main(input_data, args.normalization, args.k_values, args.synthetic_percentage)

    # Decode direction values before saving
    for key, synthetic_data in synthetic_datasets.items():
        decoded_directions = [decode_direction(int(round(val[0]))) for val in
                              synthetic_data]  # Decode back to original labels
        output_data = np.column_stack((decoded_directions, synthetic_data[:, 1]))  # Combine with packet sizes

        # Save synthetic datasets
        np.savetxt(f"{key}.csv", output_data, delimiter=',', fmt='%s', header="Direction,Packet Size (bytes)",
                   comments="")
        print(f"Saved {key}.csv")
