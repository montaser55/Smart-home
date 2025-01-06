import os
import csv
import argparse
import random
import re
import ast
import json
import time
import tracemalloc

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
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


def normalize_dataset(X, method="min_max"):
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
        "SVM": (SVC, {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "probability": [True]}),
        "k-NN": (KNeighborsClassifier, {"n_neighbors": [3, 5, 7], "metric": ["euclidean", "manhattan"]}),
        "Random Forest": (RandomForestClassifier, {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]})
    }

    predictions = {}
    confidences = {}
    runtime_memory_logs = {name: {"train": [], "test": []} for name in classifiers}

    split_idx = int(0.8 * len(X_train))
    X_train_split, X_val_split = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val_split = y_train[:split_idx], y_train[split_idx:]

    for name, (clf_class, param_grid) in classifiers.items():
        print(f"Optimizing {name}...")

        train_start_time = time.time()
        tracemalloc.start()

        best_params, best_score = manual_grid_search(
            clf_class,
            param_grid,
            X_train_split,
            y_train_split,
            X_val_split,
            y_val_split,
            scaling_method if name in ["SVM", "k-NN"] else None
        )
        print(f"Best Parameters for {name}: {best_params}, Validation Accuracy: {best_score:.4f}")
        clf = clf_class(**best_params)

        if name in ["SVM", "k-NN"]:
            normalized_X_train = normalize_dataset(X_train, scaling_method)
            clf.fit(normalized_X_train, y_train)
        else:
            clf.fit(X_train, y_train)

        train_peak_memory = tracemalloc.get_traced_memory()[1] / 1024  # Peak memory in KB
        tracemalloc.stop()
        train_runtime = time.time() - train_start_time
        runtime_memory_logs[name]["train"].append(
            {"runtime_seconds": train_runtime, "memory_peak_kb": train_peak_memory})

        test_start_time = time.time()
        tracemalloc.start()

        if name in ["SVM", "k-NN"]:
            normalized_X_test = normalize_dataset(X_test, scaling_method)
            predictions[name] = clf.predict(normalized_X_test)
            confidences[name] = clf.predict_proba(normalized_X_test)
        else:
            predictions[name] = clf.predict(X_test)
            confidences[name] = clf.predict_proba(X_test)

        test_peak_memory = tracemalloc.get_traced_memory()[1] / 1024
        tracemalloc.stop()
        test_runtime = time.time() - test_start_time
        runtime_memory_logs[name]["test"].append({"runtime_seconds": test_runtime, "memory_peak_kb": test_peak_memory})

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

    print("\n=== Individual Classifier Reports ===")
    for name in classifiers:
        print(f"Classifier: {name}")
        print(classification_report(y_test, predictions[name]))

    print("\n=== Ensemble Classifier Report ===")
    print(classification_report(y_test, ensemble_predictions))

    return predictions, ensemble_predictions, y_test, runtime_memory_logs

def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

def save_results(file_path, true_labels, individual_predictions, runtime_memory_logs):
    output_data = {
        "true_labels": [convert_to_native(label) for label in true_labels],
        "predicted_labels": {name: [convert_to_native(pred) for pred in preds] for name, preds in individual_predictions.items()},
        "runtime_memory_logs": {
            name: [convert_to_native(log) for log in logs] for name, logs in runtime_memory_logs.items()
        },
    }

    with open(file_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Results saved to {file_path}")


def plot_feature_importance_avg(avg_importances, classifier_name):
    num_features = len(avg_importances)
    indices = np.arange(num_features)

    # Sort from most important to least important
    sorted_idx = np.argsort(avg_importances)[::-1]
    sorted_importances = avg_importances[sorted_idx]

    plt.figure(figsize=(12, 6))
    plt.bar(range(num_features), sorted_importances, alpha=0.7, color='b')
    plt.title(f"Average Feature Importance - {classifier_name}")
    plt.xlabel("Feature (sorted by importance)")
    plt.ylabel("Importance Score")


    # (B) Or shift everything to be non-negative:
    min_val = sorted_importances.min()
    if min_val < 0:
        sorted_importances = sorted_importances - min_val  # shift up so min is 0
        plt.bar(range(num_features), sorted_importances, alpha=0.7, color='b')

    # Label only top 10 / bottom 10 to reduce clutter ...
    # < label logic as before >
    # Sort from most important to least important
    sorted_idx = np.argsort(avg_importances)[::-1]
    sorted_importances = avg_importances[sorted_idx]
    sorted_indices = indices[sorted_idx]

    # Label only top 10 and bottom 10 indices
    top_10_indices = sorted_indices[:10]
    bottom_10_indices = sorted_indices[-10:]
    for i in range(10):
        # label top 10
        idx_x = i
        feature_index = top_10_indices[i]
        plt.text(idx_x, sorted_importances[idx_x], str(feature_index),
                 rotation=90, ha='center', va='bottom', fontsize=8)

        # label bottom 10
        idx_x = num_features - 1 - i
        feature_index = bottom_10_indices[-1 - i]
        plt.text(idx_x, sorted_importances[idx_x], str(feature_index),
                 rotation=90, ha='center', va='bottom', fontsize=8)



    plt.tight_layout()
    plt.savefig(f"../output/{classifier_name}_avg_feature_importance.png")
    plt.close()


def train_with_subset_of_features(X_train, y_train, X_test, y_test, classifier, feature_importance, batch_size=6):
    """
    Train a classifier using subsets of features based on importance.
    Returns a list of (num_features, accuracy) pairs.
    """
    sorted_idx = np.argsort(feature_importance)[::-1]
    num_features = len(feature_importance)
    results = []

    for i in range(batch_size, num_features + 1, batch_size):
        selected_features = sorted_idx[:i]
        X_train_subset = X_train[:, selected_features]
        X_test_subset = X_test[:, selected_features]

        classifier.fit(X_train_subset, y_train)
        accuracy = classifier.score(X_test_subset, y_test)
        results.append((i, accuracy))

    return results


def analyze_feature_importance_fold(
        X_train, y_train, X_test, y_test, scaling_method, classifiers
):
    """
    Perform feature importance analysis for each classifier *on this fold only*.
    Returns a dict of:
        {
            "Random Forest": {
                "importance": <1D np.array>,
                "subset_results": <list of (num_features, accuracy) pairs>
            },
            "SVM": {...},
            "k-NN": {...}
        }
    """
    fold_results = {}

    # We’ll use normalized data for SVM and k-NN if needed
    from sklearn.feature_selection import RFE
    from sklearn.inspection import permutation_importance

    # Convert to numpy for indexing
    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    for name, clf in classifiers.items():
        print(f"\n[Fold] Analyzing feature importance for {name}...")

        if name == "Random Forest":
            # Fit on raw X
            clf.fit(X_train_np, y_train_np)
            feature_importance = clf.feature_importances_
            subset_results = train_with_subset_of_features(
                X_train_np, y_train_np, X_test_np, y_test_np, clf, feature_importance
            )

        elif name == "SVM":
            # Normalize data
            normalized_X_train = normalize_dataset(X_train, scaling_method)
            normalized_X_test = normalize_dataset(X_test, scaling_method)
            norm_X_train_np = np.array(normalized_X_train)
            norm_X_test_np = np.array(normalized_X_test)

            # Use RFE to get ranking
            # (If you want the absolute magnitude of the coefficients, you can also do that.)
            rfe = RFE(clf, n_features_to_select=1)
            rfe.fit(norm_X_train_np, y_train_np)
            # RFE gives a ranking: 1 = most important, 2 = next, etc.
            # Convert that ranking to an "importance" by taking 1/rank
            ranks = rfe.ranking_
            feature_importance = 1.0 / ranks

            subset_results = train_with_subset_of_features(
                norm_X_train_np, y_train_np, norm_X_test_np, y_test_np, clf, feature_importance
            )

        elif name == "k-NN":
            # Normalize data
            normalized_X_train = normalize_dataset(X_train, scaling_method)
            normalized_X_test = normalize_dataset(X_test, scaling_method)
            norm_X_train_np = np.array(normalized_X_train)
            norm_X_test_np = np.array(normalized_X_test)

            clf.fit(norm_X_train_np, y_train_np)
            # Use permutation_importance
            result = permutation_importance(
                clf, norm_X_train_np, y_train_np, n_repeats=5, random_state=42
            )
            feature_importance = result.importances_mean

            subset_results = train_with_subset_of_features(
                norm_X_train_np, y_train_np, norm_X_test_np, y_test_np, clf, feature_importance
            )
        else:
            # If you have more classifiers, handle them here
            feature_importance = np.zeros(X_train_np.shape[1])
            subset_results = []

        fold_results[name] = {
            "importance": feature_importance,
            "subset_results": subset_results
        }

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation for Open/Closed World Scenarios")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing CSV files.")
    parser.add_argument("--k", type=int, required=True, help="Number of folds for cross-validation.")
    parser.add_argument("--m", type=int, default=150, help="Number of sample points.")
    parser.add_argument("--n", type=int, required=False, help="Number of flows to take.")
    parser.add_argument("--scenario", type=str, choices=["closed", "open"], required=True,
                        help="Evaluation scenario: 'closed' or 'open'.")
    parser.add_argument("--foreground", type=str, required=False,
                        help="Foreground device name for the open-world scenario (e.g., 'doorsensor').")
    parser.add_argument("--ensemble_method", choices=["random", "highest_confidence", "p1_p2_diff"], default="random",
                        required=False,
                        help="Ensemble method: 'random', 'highest_confidence', or 'p1_p2_diff'.")
    parser.add_argument("--scaling_method", choices=["min_max", "z_score"], default="min_max", required=False,
                        help="Scaling method: 'min_max' or 'z_score'")
    args = parser.parse_args()

    # 1. Read dataset
    dataset = read_dataset(args.folder, args.m)
    if args.n is not None:
        dataset = truncate_dataset(dataset, args.n)

    # 2. Add labels
    labeled_dataset, label_mapping = add_label(dataset, args.scenario, args.foreground)

    # 3. Generate X, y
    X, y = generate_dataset(labeled_dataset)

    # 4. Define classifiers
    classifiers = {
        "SVM": SVC(probability=True, kernel="linear"),
        "k-NN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # 5. K-Fold Split
    k_folds = k_fold_split(X, y, args.k)

    # Prepare data structures for ACCUMULATING feature importance over folds.
    # We won't plot inside each fold. Instead, we sum them up, then plot at the end.
    num_features = len(X[0])  # Based on your code, each row is [features..., label], so check dimension carefully.
    feature_importance_sums = {
        "SVM": np.zeros(num_features),
        "k-NN": np.zeros(num_features),
        "Random Forest": np.zeros(num_features)
    }
    subset_experiment_results = {
        "SVM": [],
        "k-NN": [],
        "Random Forest": []
    }

    # 6. For each fold, train and accumulate feature importances
    for fold_index, (train_idx, test_idx) in enumerate(k_folds):
        print(f"\n[INFO] Processing Fold {fold_index + 1}/{len(k_folds)}")

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        # Analyze feature importance on this fold
        fold_results = analyze_feature_importance_fold(
            X_train, y_train, X_test, y_test, args.scaling_method, classifiers
        )

        # Accumulate importances
        for clf_name in fold_results:
            fi = fold_results[clf_name]["importance"]
            if fi.shape[0] == feature_importance_sums[clf_name].shape[0]:
                feature_importance_sums[clf_name] += fi

            # Store subset-of-features experiment results
            # (If you prefer to average them across folds or store them all, adapt as needed.)
            subset_experiment_results[clf_name].append(fold_results[clf_name]["subset_results"])

    # 7. Average feature importance and create final plots
    for clf_name in feature_importance_sums:
        # Compute average
        avg_importance = feature_importance_sums[clf_name] / args.k
        # Plot the average
        plot_feature_importance_avg(avg_importance, clf_name)

    # 8. Save subset-of-features experiments to a file and optionally plot them
    # Example: average the results across folds, then plot
    subset_experiment_averages = {}
    for clf_name, all_folds_results in subset_experiment_results.items():
        # all_folds_results is a list of lists: each item is [(n_feats, acc), (n_feats, acc), ...] for each fold
        # We want to average them by n_feats across folds.
        # One naive approach: we assume each fold used the exact same increments (6, 12, 18, ...).
        # So, we can zip them up.
        if not all_folds_results:
            continue

        # Number of folds
        fold_count = len(all_folds_results)

        # Transpose: each fold has a list of (num_feats, acc). We'll zip them:
        # e.g. [ [(6, 0.9), (12, 0.92)], [(6, 0.88), (12, 0.91)] ]
        # zipping each index across folds
        fold_0 = all_folds_results[0]
        subsets_len = len(fold_0)
        avg_results = []
        for idx_in_fold in range(subsets_len):
            # gather (num_feats, accuracy) across folds
            n_feats_vals = []
            acc_vals = []
            for fold_result_list in all_folds_results:
                # fold_result_list is something like [(6, acc1), (12, acc2), ...]
                n_feats_vals.append(fold_result_list[idx_in_fold][0])
                acc_vals.append(fold_result_list[idx_in_fold][1])
            # They should all have the same number_of_features in each fold's idx_in_fold
            num_feats = n_feats_vals[0]
            avg_accuracy = np.mean(acc_vals)
            avg_results.append((num_feats, avg_accuracy))

        subset_experiment_averages[clf_name] = avg_results

    # Save the subset experiment averages to a JSON file
    subset_results_path = "../output/subset_feature_experiment_results.json"
    with open(subset_results_path, "w") as f:
        json.dump(subset_experiment_averages, f, indent=4)
    print(f"[INFO] Subset-of-features experiment results saved to {subset_results_path}.")

    # Optionally, plot accuracy vs. #features for each classifier
    plt.figure(figsize=(8, 6))
    for clf_name, avg_results in subset_experiment_averages.items():
        x_vals = [item[0] for item in avg_results]  # number of features
        y_vals = [item[1] for item in avg_results]  # average accuracy
        plt.plot(x_vals, y_vals, marker='o', label=clf_name)
    plt.title("Accuracy vs. Number of Features (Averaged Across Folds)")
    plt.xlabel("Number of Features")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../output/subset_features_accuracy_plot.png")
    plt.close()
    print("[INFO] Subset-of-features accuracy plot saved.")


if __name__ == "__main__":
    main()