import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random

from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout, Dense, Flatten, Input, MaxPooling1D


def manhattan_distance(sample1, sample2):
    return np.sum(np.abs(sample1 - sample2))


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


def generate_synthetic_dataset(X_real, k=3):
    X_real_flat = X_real.reshape(-1, X_real.shape[-1])
    synthetic_samples = generate_synthetic_samples(X_real_flat, k=k, total_synthetic_samples=len(X_real_flat))
    synthetic_samples = synthetic_samples.reshape(X_real.shape)
    return synthetic_samples


def quote_words(line: str):
    fixed_line = re.sub(r"\bsend\b", '"send"', line)
    fixed_line = re.sub(r"\breceive\b", '"receive"', fixed_line)
    return ast.literal_eval(fixed_line)


def process_flow(data, n, include_features):
    data = data[:n]
    data = np.pad(data, (0, max(0, n - len(data))), mode='constant')

    if include_features:
        total_packets = np.count_nonzero(data)
        incoming_packets = np.sum(data < 0)
        outgoing_packets = total_packets - incoming_packets
        incoming_ratio = (incoming_packets / total_packets) if total_packets > 0 else 0
        outgoing_ratio = (outgoing_packets / total_packets) if total_packets > 0 else 0
        features = np.array([total_packets, incoming_packets, outgoing_packets, incoming_ratio, outgoing_ratio], dtype=np.float32)

        combined = []
        for i in range(n):
            row_i = np.concatenate(([data[i]], features))
            combined.append(row_i)
        return np.array(combined, dtype=np.float32)
    else:
        return data.reshape(n, 1).astype(np.float32)



def load_data(main_folder, m, n, include_features):
    dataset = {}
    device_to_class = {}
    class_idx = 0

    for file_name in os.listdir(main_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(main_folder, file_name)

            device_name = file_name.split('_')[0]
            device_to_class[device_name] = class_idx
            class_idx += 1

            df = pd.read_csv(file_path, header=None, skiprows=1, sep=';', engine='python')
            num_flows = min(m, len(df))

            all_flows_for_device = []
            for i in range(num_flows):
                row_str = df.iloc[i, 0]
                packet_list = quote_words(row_str)

                flow_data = []
                for direction, size in packet_list:
                    flow_data.append(-float(size) if direction == 'send' else float(size))

                flow_matrix = process_flow(np.array(flow_data, dtype=np.float32), n, include_features)
                all_flows_for_device.append(flow_matrix)

            dataset[device_name] = all_flows_for_device
    return dataset


def build_cnn(input_shape, num_classes, kernel_size, dropout_rate):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in [64, 128, 256, 512]:
        model.add(Conv1D(i, kernel_size=kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))

        model.add(Conv1D(i, kernel_size=kernel_size, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(dropout_rate))

        model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision', class_id=1),
            tf.keras.metrics.Recall(name='recall', class_id=1)
        ]
    )
    return model



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


def tune_hyperparameters(X_train, y_train, X_val, y_val, epochs):
    learning_rates = [1e-3, 1e-4]
    dropout_rates = [0.3, 0.4]
    kernel_sizes = [3, 5]
    batch_sizes = [16, 32]

    best_accuracy = 0
    best_params = None

    X_train = np.tile(X_train, (3, 1, 1))
    y_train = np.tile(y_train, (3, 1))

    for lr in learning_rates:
        for dr in dropout_rates:
            for ks in kernel_sizes:
                for bs in batch_sizes:
                    model = build_cnn(X_train.shape[1:], y_train.shape[1], ks, dr)
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                  loss='categorical_crossentropy', metrics=['accuracy'])
                    history = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_data=(X_val, y_val), verbose=0)
                    val_accuracy = np.mean(history.history['val_accuracy'])
                    print(f"Learning Rate: {lr}, Dropout Rate: {dr}, Kernel Size: {ks}, Batch Size: {bs} == Average Validation Accuracy: {val_accuracy:.4f}")
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = (lr, dr, ks, bs)

    print("\nBest Hyperparameters:")
    print(f"Learning Rate: {best_params[0]}, Dropout Rate: {best_params[1]}, Kernel Size: {best_params[2]}, Batch Size: {best_params[3]}")
    print(f"Best Average Validation Accuracy: {best_accuracy:.4f}")
    return best_params


def replace_with_synthetic_data(X_train, y_train, k=3):
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    unique_labels = np.unique(y_train)

    X_synthetic_list = []
    y_synthetic_list = []

    for label in unique_labels:
        indices = np.where(y_train == label)[0]
        X_label = X_train[indices]

        X_label_synth = generate_synthetic_dataset(X_label, k=k)
        X_synthetic_list.append(X_label_synth)
        y_synthetic_list.append(np.full(len(X_label_synth), label))

    X_synthetic = np.concatenate(X_synthetic_list, axis=0)
    y_synthetic = np.concatenate(y_synthetic_list, axis=0)

    idx = np.arange(len(X_synthetic))
    np.random.shuffle(idx)
    X_synthetic = X_synthetic[idx]
    y_synthetic = y_synthetic[idx]

    return X_synthetic, y_synthetic


def train_and_evaluate_kfold(dataset, foreground_device, best_params, k, epochs, val_split, k_synth):

    fold_metrics = []
    best_lr, best_dr, best_ks, best_bs = best_params
    k_splits = open_world_k_fold_split(dataset, foreground_device, k)

    for fold_idx, (X_train_list, y_train_list, X_test_list, y_test_list) in enumerate(k_splits, 1):
        print(f"\n=== Fold {fold_idx}/{len(k_splits)} ===")

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.int32)
        X_test = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list, dtype=np.int32)

        print("Generating synthetic training data")
        X_train, y_train = replace_with_synthetic_data(X_train, y_train, k=k_synth)

        if X_train.ndim == 2:
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)

        y_train = tf.keras.utils.to_categorical(y_train, 2).astype(np.float32)

        X_train = np.tile(X_train, (3, 1, 1))
        y_train = np.tile(y_train, (3, 1))

        model = build_cnn(X_train.shape[1:], 2, best_ks, best_dr)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision', class_id=1),
                tf.keras.metrics.Recall(name='recall', class_id=1)
            ]
        )

        model.fit(X_train, y_train, epochs=epochs, batch_size=best_bs,
                  validation_split=val_split, verbose=1)

        metrics = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test, 2).astype(np.float32), verbose=0)

        fold_accuracy = metrics[1]
        fold_precision = metrics[2]
        fold_recall = metrics[3]

        fold_metrics.append((fold_precision, fold_recall))

        print(f"Fold Accuracy: {fold_accuracy:.4f}, Fold Precision: {fold_precision:.4f}, Fold Recall: {fold_recall:.4f}")

    avg_precision = np.mean([m[0] for m in fold_metrics])
    avg_recall = np.mean([m[1] for m in fold_metrics])

    return avg_precision, avg_recall


def train_val_split(all_X, all_y, val_split):
    class_to_flows = defaultdict(list)

    for flow, label in zip(all_X, all_y):
        class_to_flows[label].append(flow)

    X_train, X_val = [], []
    y_train, y_val = [], []

    for class_label, flows in class_to_flows.items():
        random.shuffle(flows)
        num_val_samples = int(len(flows) * val_split)

        val_samples = flows[:num_val_samples]
        train_samples = flows[num_val_samples:]

        X_val.extend(val_samples)
        y_val.extend([class_label] * len(val_samples))
        X_train.extend(train_samples)
        y_train.extend([class_label] * len(train_samples))

    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_val = np.array(y_val, dtype=np.int32)

    y_train = tf.keras.utils.to_categorical(y_train, 2).astype(np.float32)
    y_val = tf.keras.utils.to_categorical(y_val, 2).astype(np.float32)

    return X_train, X_val, y_train, y_val


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning and k-fold evaluation for CNN with synthetic data (open world)")
    parser.add_argument("--main_folder", type=str, required=True, help="Folder containing CSV files.")
    parser.add_argument("--m", type=int, default=45, help="Number of flows to take from each device")
    parser.add_argument("--n", type=int, default=100, help="Number of send/receives per flow")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--k", type=int, default=5, help="Number of folds for k-fold cross-validation")
    parser.add_argument("--output_plot", type=str, default="precision_recall_synthetic_plot.png", help="Filename to save the plot")
    args = parser.parse_args()

    results = []

    for include_features in [True, False]:
        print(f"include_features={include_features}")

        dataset = load_data(args.main_folder, args.m, args.n, include_features)
        print("Data loaded")

        for foreground_device in dataset.keys():
            print(f"\nForeground Device: {foreground_device}")
            all_X = []
            all_y = []

            for device_name, flows in dataset.items():
                label = 0 if device_name == foreground_device else 1
                for fm in flows:
                    all_X.append(fm)
                    all_y.append(label)

            all_X = np.array(all_X, dtype=np.float32)
            all_y = np.array(all_y, dtype=np.int32)

            X_train, X_val, y_train, y_val = train_val_split(all_X, all_y, args.val_split)

            print("Starting hyperparameter tuning")
            best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, args.epochs)

            avg_precision, avg_recall = train_and_evaluate_kfold(
                dataset, foreground_device, best_params, args.k, args.epochs, args.val_split, 3
            )

            results.append({
                "Foreground Device": foreground_device,
                "Include Features": include_features,
                "Avg Precision": avg_precision,
                "Avg Recall": avg_recall
            })

    devices = list(set(result["Foreground Device"] for result in results))
    include_features = [True, False]

    fig, (ax_precision, ax_recall) = plt.subplots(1, 2, figsize=(16, 6))

    for include_feature in include_features:
        precision_values = []
        for result in results:
            if result["Include Features"] == include_feature:
                precision_values.append(result["Avg Precision"])

        ax_precision.bar(
            devices,
            precision_values,
            width=0.4,
            align="center" if include_feature else "edge",
            label=f"Precision (Features={include_feature})"
        )

    ax_precision.set_ylabel("Scores")
    ax_precision.set_title("Precision")
    ax_precision.set_xticks(range(len(devices)))
    ax_precision.set_xticklabels(devices)
    ax_precision.set_ylim(0, 1.1)
    ax_precision.legend()

    for include_feature in include_features:
        recall_values = []
        for result in results:
            if result["Include Features"] == include_feature:
                recall_values.append(result["Avg Recall"])

        ax_recall.bar(
            devices,
            recall_values,
            width=0.4,
            align="center" if include_feature else "edge",
            label=f"Recall (Features={include_feature})"
        )

    ax_recall.set_ylabel("Scores")
    ax_recall.set_title("Recall")
    ax_recall.set_xticks(range(len(devices)))
    ax_recall.set_xticklabels(devices)
    ax_recall.set_ylim(0, 1.1)
    ax_recall.legend()

    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Precision Recall (Synthetic) plot saved as {args.output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
