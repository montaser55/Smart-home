import os
import csv
import argparse
import re
import ast
import json

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
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
        start = 0
        for i in range(k_folds):
            end = start + fold_size + (1 if i < remainder else 0)
            folds[i].extend(indices[start:end])
            start = end

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


def plot_feature_importance_avg(avg_importances, classifier_name, output_dir = "../output", top_n=30):
    sorted_idx = np.argsort(avg_importances)[::-1]
    sorted_vals = avg_importances[sorted_idx]
    min_val = sorted_vals.min()
    if min_val < 0:
        sorted_vals = sorted_vals - min_val

    top_indices = sorted_idx[:top_n]
    top_importances = sorted_vals[:top_n]

    bottom_indices = sorted_idx[-top_n:]
    bottom_importances = sorted_vals[-top_n:]

    combined_indices = np.concatenate([top_indices, bottom_indices])
    combined_importances = np.concatenate([top_importances, bottom_importances])

    plt.figure(figsize=(12, 6))
    x_positions = np.arange(len(combined_importances))
    plt.bar(x_positions, combined_importances, alpha=0.7, color='b')
    plt.title(f"Top {top_n} & Bottom {top_n} Feature Importances - {classifier_name}", fontsize=14)
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Importance Score", fontsize=12)

    for i, feat_idx in enumerate(combined_indices):
        plt.text(
            i,
            combined_importances[i],
            str(feat_idx),
            ha='center',
            va='bottom',
            rotation=90,
            fontsize=9
        )

    plt.tight_layout()

    out_file = os.path.join(output_dir, f"{classifier_name}_top_bottom_{top_n}.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved top-bottom-{top_n} plot to '{out_file}'.")


def train_with_subset_of_features(X_train, y_train, X_test, y_test, classifier, feature_importance, batch_size=6):
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


def analyze_feature_importance_fold( X_train, y_train, X_test, y_test, scaling_method, classifiers):
    fold_results = {}

    X_train_np = np.array(X_train)
    X_test_np = np.array(X_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    for name, clf in classifiers.items():
        print(f"\n[Fold] Analyzing feature importance for {name}...")

        if name == "Random Forest":
            clf.fit(X_train_np, y_train_np)
            feature_importance = clf.feature_importances_
            subset_results = train_with_subset_of_features(
                X_train_np, y_train_np, X_test_np, y_test_np, clf, feature_importance
            )

        elif name == "SVM":
            normalized_X_train = normalize_dataset(X_train, scaling_method)
            normalized_X_test = normalize_dataset(X_test, scaling_method)
            norm_X_train_np = np.array(normalized_X_train)
            norm_X_test_np = np.array(normalized_X_test)
            rfe = RFE(clf, n_features_to_select=1)
            rfe.fit(norm_X_train_np, y_train_np)
            ranks = rfe.ranking_
            feature_importance = 1.0 / ranks

            subset_results = train_with_subset_of_features(
                norm_X_train_np, y_train_np, norm_X_test_np, y_test_np, clf, feature_importance
            )

        elif name == "k-NN":
            normalized_X_train = normalize_dataset(X_train, scaling_method)
            normalized_X_test = normalize_dataset(X_test, scaling_method)
            norm_X_train_np = np.array(normalized_X_train)
            norm_X_test_np = np.array(normalized_X_test)

            clf.fit(norm_X_train_np, y_train_np)
            result = permutation_importance(
                clf, norm_X_train_np, y_train_np, n_repeats=5, random_state=42
            )
            feature_importance = result.importances_mean

            subset_results = train_with_subset_of_features(norm_X_train_np, y_train_np, norm_X_test_np, y_test_np, clf, feature_importance)

        fold_results[name] = {
            "importance": feature_importance,
            "subset_results": subset_results
        }

    return fold_results


def plt_accuracy_vs_num_features(subset_experiment_averages, output_dir = "../output"):
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
    output_path = os.path.join(output_dir, f"subset_features_accuracy_plot.png")
    plt.savefig(output_path)
    plt.close()
    print("Subset-of-features accuracy plot saved.")


def generate_subset_experiments_averages(subset_experiment_results, output_dir = "../output"):
    subset_experiment_averages = {}
    for clf_name, all_folds_results in subset_experiment_results.items():
        if not all_folds_results:
            continue

        fold_0 = all_folds_results[0]
        subsets_len = len(fold_0)
        avg_results = []
        for idx_in_fold in range(subsets_len):
            n_feats_vals = []
            acc_vals = []
            for fold_result_list in all_folds_results:
                n_feats_vals.append(fold_result_list[idx_in_fold][0])
                acc_vals.append(fold_result_list[idx_in_fold][1])
            num_feats = n_feats_vals[0]
            avg_accuracy = np.mean(acc_vals)
            avg_results.append((num_feats, avg_accuracy))

        subset_experiment_averages[clf_name] = avg_results

    subset_results_path = os.path.join(output_dir, "subset_feature_experiment_results.json")
    with open(subset_results_path, "w") as f:
        json.dump(subset_experiment_averages, f, indent=4)
    print(f"Subset-of-features experiment results saved to {subset_results_path}.")

    return subset_experiment_averages


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

    dataset = read_dataset(args.folder, args.m)
    if args.n is not None:
        dataset = truncate_dataset(dataset, args.n)
    labeled_dataset, label_mapping = add_label(dataset, args.scenario, args.foreground)
    X, y = generate_dataset(labeled_dataset)

    classifiers = {
        "SVM": SVC(probability=True, kernel="linear"),
        "k-NN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    k_folds = k_fold_split(X, y, args.k)

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

    for fold_index, (train_idx, test_idx) in enumerate(k_folds):
        print(f"\nProcessing Fold {fold_index + 1}/{len(k_folds)}")

        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train = [y[i] for i in train_idx]
        y_test = [y[i] for i in test_idx]

        fold_results = analyze_feature_importance_fold(X_train, y_train, X_test, y_test, args.scaling_method, classifiers)

        for clf_name in fold_results:
            fi = fold_results[clf_name]["importance"]
            if fi.shape[0] == feature_importance_sums[clf_name].shape[0]:
                feature_importance_sums[clf_name] += fi
            subset_experiment_results[clf_name].append(fold_results[clf_name]["subset_results"])

    for clf_name in feature_importance_sums:
        avg_importance = feature_importance_sums[clf_name] / args.k
        plot_feature_importance_avg(avg_importance, clf_name)

    subset_experiment_averages = generate_subset_experiments_averages(subset_experiment_results)
    plt_accuracy_vs_num_features(subset_experiment_averages)


if __name__ == "__main__":
    main()