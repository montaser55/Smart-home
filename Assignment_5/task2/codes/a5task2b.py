import os
import csv
import argparse
import random
import re
import ast

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


def parse_flow_string(flow_string):
    direction_words = ['receive', 'incoming', 'send', 'outgoing']
    pattern = r'(\W)(' + '|'.join(direction_words) + r')(\W)'
    replacement = r'\1"\2"\3'

    corrected = re.sub(pattern, replacement, flow_string)
    return ast.literal_eval(corrected)


def compute_stats_and_abs_cumulative_sums(flow_pairs):
    incoming_packets = 0
    outgoing_packets = 0
    incoming_size_sum = 0
    outgoing_size_sum = 0

    cumulative_sums = []
    running_sum = 0

    for direction_str, size in flow_pairs:
        direction = direction_str.lower().strip()

        if direction in ['receive', 'incoming']:
            incoming_packets += 1
            incoming_size_sum += size
        else:
            outgoing_packets += 1
            outgoing_size_sum += size

        running_sum += size
        cumulative_sums.append(running_sum)

    stats = {
        'incoming_packets': incoming_packets,
        'outgoing_packets': outgoing_packets,
        'incoming_size_sum': incoming_size_sum,
        'outgoing_size_sum': outgoing_size_sum
    }
    return stats, cumulative_sums

def sample_cumulative_sums(cumulative_sums, m):
    if not cumulative_sums:
        return [0] * m

    indices = np.linspace(0, len(cumulative_sums) - 1, m).astype(int)
    return [cumulative_sums[i] for i in indices]

def transform_data(input_string):
    def replacer(match):
        direction, value = match.groups()
        value = int(value)
        return f"{-value}" if direction == "receive" else f"{value}"

    transformed_string = re.sub(r"\[(receive|send),\s*(\d+)\]", replacer, input_string)
    return transformed_string

def generate_features(flow_string, m):
    flow_pairs = parse_flow_string(flow_string)
    stats, abs_sums = compute_stats_and_abs_cumulative_sums(flow_pairs)
    sampled = sample_cumulative_sums(abs_sums, m)

    stats_values = list(stats.values())
    combined_list = stats_values + sampled
    return str(combined_list)


def read_dataset(folder_path, m):
    dataset = {}
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            label = os.path.splitext(file)[0].split("_")[0]
            with open(os.path.join(folder_path, file), 'r') as f:
                reader = csv.reader(f)
                next(reader)

                data = []
                for row in reader:
                    row_str = ','.join(row).strip()
                    row_str = generate_features(row_str, m)


                    if row_str.startswith("[") and row_str.endswith("]"):
                        try:
                            parsed_row = eval(row_str)
                            data.append(parsed_row)
                        except SyntaxError as e:
                            print(f"Error parsing row in file {file}: {row_str}")
                            raise e
                dataset[label] = data
    return dataset



def add_label(dataset, scenario, foreground_devices=None):
    labeled_data = []

    if scenario == "closed":
        device_names = list(dataset.keys())
        class_label_mapping = {device: idx for idx, device in enumerate(device_names)}

        for device, packets in dataset.items():
            label = class_label_mapping[device]
            for packet in packets:
                labeled_data.append(packet + [label])

    elif scenario == "open":
        if not foreground_devices:
            raise ValueError("Foreground devices must be specified for the open-world scenario.")

        class_label_mapping = {"foreground": 0, "background": 1}

        for device, packets in dataset.items():
            label = 0 if device in foreground_devices else 1
            for packet in packets:
                labeled_data.append(packet + [label])

    else:
        raise ValueError("Unsupported scenario. Use 'closed' or 'open'.")

    return labeled_data, class_label_mapping



def generate_dataset(dataset):
    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]
    return X, y


def k_fold_split(X, y, k_folds):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y):
        label_to_indices[label].append(idx)

    for label in label_to_indices:
        np.random.shuffle(label_to_indices[label])

    folds = [[] for _ in range(k_folds)]
    for label, indices in label_to_indices.items():
        fold_size = len(indices) // k_folds
        remainder = len(indices) % k_folds
        print(f'len(indices): {len(indices)}, fold_size: {fold_size}, remainder: {remainder}')
        start = 0
        for i in range(k_folds):
            end = start + fold_size + (1 if i < remainder else 0)
            folds[i].extend(indices[start:end])
            print(f'start: {start}, end: {end}')
            start = end
    print(f'folds: {folds}')

    train_test_splits = []
    for i in range(k_folds):
        test_idx = folds[i]
        train_idx = [idx for j in range(k_folds) if j != i for idx in folds[j]]
        train_test_splits.append((train_idx, test_idx))
    print(f'train_test_splits: {train_test_splits}')
    return train_test_splits

def manual_grid_search(classifier, param_grid, X_train, y_train, X_val, y_val, scaling_method=None):
    best_params = None
    best_score = -1

    from itertools import product
    param_combinations = list(product(*param_grid.values()))

    if scaling_method:
        X_train_normalized = normalize_dataset(X_train, scaling_method)
        X_val_normalized = normalize_dataset(X_val, scaling_method)
    else:
        X_train_normalized, X_val_normalized = X_train, X_val

    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))
        clf = classifier(**param_dict)

        clf.fit(X_train_normalized, y_train)
        val_predictions = clf.predict(X_val_normalized)
        val_score = np.mean(val_predictions == y_val)

        if val_score > best_score:
            best_score = val_score
            best_params = param_dict

    return best_params, best_score

def min_max_normal(X):
    flat_X = np.concatenate([np.array(row) for row in X])
    global_min = flat_X.min()
    global_max = flat_X.max()
    return [(np.array(row) - global_min) / (global_max - global_min) for row in X]

def z_score_normal(X):
    flat_X = np.concatenate([np.array(row) for row in X])
    global_mean = flat_X.mean()
    global_std = flat_X.std()
    return [(np.array(row) - global_mean) / global_std for row in X]


def normalize_dataset(X, method):
    if method == "min_max":
        return min_max_normal(X)
    elif method == "z_score":
        return z_score_normal(X)
    else:
        raise ValueError("Unsupported scaling method. Use 'min_max' or 'z_score'.")


def truncate_dataset(dataset, n):
    truncated_dataset = {}
    for key, value_list in dataset.items():
        truncated_dataset[key] = value_list[:n]
    return truncated_dataset

def open_world_k_fold_split(dataset, foreground_device, k_folds):
    foreground_data = dataset.get(foreground_device)
    random.shuffle(foreground_data)
    background_devices = list(dataset.keys())
    random.shuffle(background_devices)

    fg_fold_size = len(foreground_data) // k_folds
    remainder = len(foreground_data) % k_folds
    foreground_folds = [
        foreground_data[i * fg_fold_size + min(i, remainder): (i + 1) * fg_fold_size + min(i + 1, remainder)]
        for i in range(k_folds)
    ]

    background_folds = []
    num_background_devices = len(background_devices)
    for i in range(k_folds):
        test_device = background_devices[i % num_background_devices]
        train_devices = [device for device in background_devices if device != test_device]
        train_data = [packet for device in train_devices for packet in dataset[device]]
        test_data = dataset[test_device]
        background_folds.append((train_data, test_data))


    combined_folds = []
    for i in range(k_folds):
        fg_test = foreground_folds[i]
        fg_train = [
            packet for j, fold in enumerate(foreground_folds) if j != i for packet in fold
        ]
        bg_train, bg_test = background_folds[i]

        X_train = fg_train + bg_train
        X_test = fg_test + bg_test

        y_train = [0] * len(fg_train) + [1] * len(bg_train)
        y_test = [0] * len(fg_test) + [1] * len(bg_test)

        combined_folds.append((X_train, y_train, X_test, y_test))

    return combined_folds

def min_max_normalization(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    if min_vals == max_vals:
        normalized_data = np.zeros_like(data)
    else:
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


def restore_to_original_structure(arrays):
    return [array.flatten().tolist() for array in arrays]

def find_k_nearest_neighbors(data, sample, k):
    distances = [(i, manhattan_distance(sample, other_sample)) for i, other_sample in enumerate(data) if
                 not np.array_equal(sample, other_sample)]
    distances.sort(key=lambda x: x[1])
    return [data[i] for i, _ in distances[:k]]


def generate_synthetic_samples(data, k, total_synthetic_samples):
    synthetic_data = []
    samples_to_generate = total_synthetic_samples // len(data)
    remaining_samples = total_synthetic_samples % len(data)

    for i, sample in enumerate(data):
        k_neighbors = find_k_nearest_neighbors(data, sample, k)
        if len(k_neighbors) == 0:
            k_neighbors = [sample]
        for _ in range(samples_to_generate):
            neighbor = random.choice(k_neighbors)
            diff = neighbor - sample
            random_scale = random.uniform(0, 1)
            synthetic_sample = sample + random_scale * diff
            synthetic_data.append(synthetic_sample)

        if i < remaining_samples:
            neighbor = random.choice(k_neighbors)
            diff = neighbor - sample
            random_scale = random.uniform(0, 1)
            synthetic_sample = sample + random_scale * diff
            synthetic_data.append(synthetic_sample)

    return np.array(synthetic_data)


def generate_data(input_data, k=3, synthetic_sample_percentage=100, normalization_method="min_max"):
    features = input_data[:4]
    input_data = input_data[4:]

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

    total_synthetic_samples = int(synthetic_sample_percentage * len(data) / 100)
    synthetic_data = generate_synthetic_samples(data, k, total_synthetic_samples)
    synthetic_dataset = denormalize_fn(synthetic_data, param1, param2)
    synthetic_dataset = np.vstack((features, synthetic_dataset))

    return synthetic_dataset


def transform_to_2d_arrays(X):
    transformed_data = [np.array(sublist, dtype=float).reshape(-1, 1) for sublist in X]
    return transformed_data


def create_synthetic_dataset(dataset):
    transformed_dataset = transform_to_2d_arrays(dataset)
    synthetic_dataset = []
    for _, flow in enumerate(transformed_dataset):
        synthetic_flow = generate_data(flow)
        synthetic_dataset.append(synthetic_flow)
    return restore_to_original_structure(synthetic_dataset)


def train_test_models(X_train, y_train, X_test, y_test, ensemble_method, scaling_method):
    X_train = create_synthetic_dataset(X_train)
    classifiers = {
        "SVM": (SVC(probability=True), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "k-NN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7], "metric": ["euclidean", "manhattan"]}),
        "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]})
    }

    predictions = {}
    confidences = {}
    runtime_memory_logs = {name: {"train": [], "test": []} for name in classifiers}

    split_idx = int(0.8 * len(X_train))
    X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

    for name, (clf, param_grid) in classifiers.items():
        print(f"Optimizing {name} with GridSearchCV...")

        if name in ["SVM", "k-NN"]:
            X_train_normalized = normalize_dataset(X_train_split, scaling_method)
            grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=3)
            grid_search.fit(X_train_normalized, y_train_split)
            best_clf = grid_search.best_estimator_

            normalized_X_train = normalize_dataset(X_train, scaling_method)
            normalized_X_test = normalize_dataset(X_test, scaling_method)
            best_clf.fit(normalized_X_train, y_train)
            predictions[name] = best_clf.predict(normalized_X_test)
            confidences[name] = best_clf.predict_proba(normalized_X_test)
        else:
            grid_search = GridSearchCV(clf, param_grid, scoring='accuracy', cv=3)
            grid_search.fit(X_train_split, y_train_split)
            best_clf = grid_search.best_estimator_
            best_clf.fit(X_train, y_train)
            predictions[name] = best_clf.predict(X_test)
            confidences[name] = best_clf.predict_proba(X_test)

        print(f"Best Parameters for {name}: {grid_search.best_params_}")


    ensemble_predictions = []
    for i in range(len(X_test)):
        if ensemble_method == "random":
            chosen_classifier = random.choice(list(classifiers.keys()))
            ensemble_predictions.append(predictions[chosen_classifier][i])
        elif ensemble_method == "highest_confidence":
            highest_confidence_clf = max(confidences, key=lambda clf: np.max(confidences[clf][i]))
            ensemble_predictions.append(predictions[highest_confidence_clf][i])
        elif ensemble_method == "p1_p2_diff":
            max_diff = -1
            chosen_class = None
            for clf, prob in confidences.items():
                sorted_prob = np.sort(prob[i])[::-1]
                diff = sorted_prob[0] - sorted_prob[1]
                if diff > max_diff:
                    max_diff = diff
                    chosen_class = predictions[clf][i]
            ensemble_predictions.append(chosen_class)

    return predictions, ensemble_predictions, y_test, runtime_memory_logs


def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


def plot_classifier(data, output_dir="../output"):

    devices = list(data.keys())
    classifiers = list(data[devices[0]].keys())
    metrics = ["precision", "recall"]

    x = np.arange(len(classifiers))
    width = 0.15
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for metric in metrics:
        plt.figure(figsize=(12, 7))
        for i, device in enumerate(devices):
            values = [data[device][clf][metric] for clf in classifiers]
            plt.bar(x + i * width, values, width, label=device, color=colors[i % len(colors)])

        plt.xticks(x + (len(devices) - 1) * width / 2, classifiers)
        plt.title(f"{metric.capitalize()} for Different Devices and Classifiers")
        plt.xlabel("Classifiers")
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1)
        plt.legend(title="Devices")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"{metric}_plot_synthetic_data.png")
        plt.savefig(output_path)
        print(f"[INFO] {metric.capitalize()} plot saved at {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Open-World Scenario for K-Fold Cross-Validation")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing CSV files.")
    parser.add_argument("--k", type=int, required=True, help="Number of folds for cross-validation.")
    parser.add_argument("--n", type=int, required=False, help="Number of flows to take.")
    parser.add_argument("--m", type=int, default=150, help="Number of sample points.")
    parser.add_argument("--ensemble_method", choices=["random", "highest_confidence", "p1_p2_diff"], default="random",
                        help="Ensemble method: 'random', 'highest_confidence', or 'p1_p2_diff'.")
    parser.add_argument("--scaling_method", choices=["min_max", "z_score"], default="min_max",
                        help="Scaling method: 'min_max' or 'z_score'")
    args = parser.parse_args()

    dataset = read_dataset(args.folder, args.m)
    if args.n is not None:
       dataset = truncate_dataset(dataset, args.n)

    results = {}
    for foreground_device in dataset.keys():
        print(f"\n[INFO] Using {foreground_device} as the foreground device.")
        all_true_labels = []
        all_ensemble_predictions = []
        all_individual_predictions = {name: [] for name in ["SVM", "k-NN", "Random Forest"]}

        folds = open_world_k_fold_split(dataset, foreground_device, args.k)

        for fold_index, (X_train, y_train, X_test, y_test) in enumerate(folds):
            print(f"\n[INFO] Processing Fold {fold_index + 1}/{len(folds)} for {foreground_device}.")

            predictions, ensemble_predictions, true_labels, _ = train_test_models(
                X_train, y_train, X_test, y_test, args.ensemble_method, args.scaling_method
            )

            all_true_labels.extend(true_labels)
            all_ensemble_predictions.extend(ensemble_predictions)

            for name in predictions:
                all_individual_predictions[name].extend(predictions[name])

        results[foreground_device] = {}

        for name, pred_list in all_individual_predictions.items():
            precision = precision_score(all_true_labels, pred_list, average="binary")
            recall = recall_score(all_true_labels, pred_list, average="binary")
            results[foreground_device][name] = {
                "precision": precision,
                "recall": recall,
            }
            print(f"\nClassifier: {name}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")

        ensemble_precision = precision_score(all_true_labels, all_ensemble_predictions, average="binary")
        ensemble_recall = recall_score(all_true_labels, all_ensemble_predictions, average="binary")
        results[foreground_device]["ensemble"] = {
            "precision": ensemble_precision,
            "recall": ensemble_recall,
        }
        print(f"\nClassifier: Ensemble")
        print(f"Precision: {ensemble_precision:.2f}")
        print(f"Recall: {ensemble_recall:.2f}")
        print("------------------------------")

    plot_classifier(results)

if __name__ == "__main__":
    main()