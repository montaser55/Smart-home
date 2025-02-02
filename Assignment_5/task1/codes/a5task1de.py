import os
import csv
import argparse
import random
import re
import ast

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
from collections import defaultdict


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



def add_label(dataset):
    labeled_data = []

    device_names = list(dataset.keys())
    class_label_mapping = {device: idx for idx, device in enumerate(device_names)}

    for device, packets in dataset.items():
        label = class_label_mapping[device]
        for packet in packets:
            labeled_data.append(packet + [label])


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


def train_test_models(X_train, y_train, X_test, y_test, ensemble_method, scaling_method):
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

def save_file(results, n, m, output_dir = "../output/taskde/"):
    output_file = os.path.join(output_dir, f"1de_n{n}_m{m}.txt")

    with open(output_file, "w") as file:
        file.write("Classifier Results:\n")
        for classifier, metrics in results.items():
            file.write(f"\nClassifier: {classifier}\n")
            for metric, value in metrics.items():
                file.write(f"{metric.capitalize()}: {value:.2f}\n")

    print(f"[INFO] Results saved to {output_file}")

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

    labeled_dataset, label_mapping = add_label(dataset)

    X, y = generate_dataset(labeled_dataset)
    folds = k_fold_split(X, y, args.k)

    results = {}

    all_true_labels = []
    all_ensemble_predictions = []
    all_individual_predictions = {name: [] for name in ["SVM", "k-NN", "Random Forest"]}

    for fold_index, (train_idx, test_idx) in enumerate(folds):
        print(f"Processing Fold {fold_index + 1}/{len(folds)}")

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        predictions, ensemble_predictions, true_labels, _ = train_test_models(
            X_train, y_train, X_test, y_test, args.ensemble_method, args.scaling_method
        )

        all_true_labels.extend(true_labels)
        all_ensemble_predictions.extend(ensemble_predictions)

        for name in predictions:
            all_individual_predictions[name].extend(predictions[name])


    for name, pred_list in all_individual_predictions.items():
        accuracy = accuracy_score(all_true_labels, pred_list, normalize=True)
        results[name] = {
            "accuracy": accuracy,
        }
        print(f"\nClassifier: {name}")
        print(f"Accuracy: {accuracy:.2f}")

    ensemble_accuracy = accuracy_score(all_true_labels, all_ensemble_predictions, normalize=True)
    results["ensemble"] = {
        "accuracy": ensemble_accuracy
    }
    print(f"\nClassifier: Ensemble")
    print(f"Accuracy: {ensemble_accuracy:.2f}")
    print("------------------------------")

    save_file(results, args.n, args.m)

if __name__ == "__main__":
    main()