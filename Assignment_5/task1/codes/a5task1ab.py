import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dropout, Dense, Flatten, Input, MaxPooling1D


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
    flows = []
    labels = []
    device_to_class = {}
    class_idx = 0

    for file_name in os.listdir(main_folder):
        if file_name.endswith('.csv'):
            file_path = main_folder + "/" + file_name

            device_name = os.path.splitext(file_name)[0]
            device_to_class[device_name] = class_idx
            class_idx += 1

            df = pd.read_csv(file_path, header=None, skiprows=1, sep=';')
            num_flows = min(m, len(df))

            for i in range(num_flows):
                row_str = df.iloc[i, 0]
                packet_list = quote_words(row_str)

                flow_data = []
                for direction, size in packet_list:
                    flow_data.append(-float(size) if direction == 'send' else float(size))

                flow_matrix = process_flow(np.array(flow_data, dtype=np.float32), n, include_features)
                flows.append(flow_matrix)
                labels.append(device_to_class[device_name])
    flows = np.array(flows, dtype=np.float32)
    labels = tf.keras.utils.to_categorical(labels, num_classes=class_idx).astype(np.float32)

    return flows, labels, device_to_class


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

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def k_fold_split(X, y, k):
    y_int = np.argmax(y, axis=1)
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y_int):
        label_to_indices[label].append(idx)

    for label in label_to_indices:
        np.random.shuffle(label_to_indices[label])

    folds = [[] for _ in range(k)]
    for label, indices in label_to_indices.items():
        fold_size = len(indices) // k
        remainder = len(indices) % k
        start = 0
        for i in range(k):
            end = start + fold_size + (1 if i < remainder else 0)
            folds[i].extend(indices[start:end])
            start = end

    train_test_splits = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]
        train_test_splits.append((train_idx, test_idx))
    return train_test_splits


def tune_hyperparameters(X_train, y_train, X_val, y_val, epochs):
    learning_rates = [1e-3, 1e-4]
    dropout_rates = [0.3, 0.4]
    kernel_sizes = [3, 5]
    batch_sizes = [16, 32]

    best_accuracy = 0
    best_params = None

    X_train = np.tile(X_train, (6, 1, 1))
    y_train = np.tile(y_train, (6, 1))

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


def train_and_evaluate_kfold(flows, labels, best_params, k, epochs, val_split):

    y_true_all, y_pred_all = [], []
    best_lr, best_dr, best_ks, best_bs = best_params

    k_splits = k_fold_split(flows, labels, k=k)

    for fold_idx, (train_idx, test_idx) in enumerate(k_splits, 1):
        print(f"\n=== Fold {fold_idx}/{len(k_splits)} ===")

        X_train = flows[train_idx]
        y_train = labels[train_idx]
        X_test = flows[test_idx]
        y_test = labels[test_idx]

        X_train = np.tile(X_train, (6, 1, 1))
        y_train = np.tile(y_train, (6, 1))

        model = build_cnn(X_train.shape[1:], labels.shape[1], best_ks, best_dr)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=epochs, batch_size=best_bs, validation_split=val_split, verbose=1)

        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

    return y_true_all, y_pred_all


def train_val_split(flows, labels, val_split):
    class_to_flows = defaultdict(list)
    for flow, label in zip(flows, labels):
        class_idx = np.argmax(label)
        class_to_flows[class_idx].append((flow, label))

    X_train, X_val = [], []
    y_train, y_val = [], []

    for class_idx, samples in class_to_flows.items():
        np.random.shuffle(samples)
        num_val_samples = int(len(samples) * val_split)

        val_samples = samples[:num_val_samples]
        train_samples = samples[num_val_samples:]

        for flow, label in val_samples:
            X_val.append(flow)
            y_val.append(label)

        for flow, label in train_samples:
            X_train.append(flow)
            y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)

    return X_train, X_val, y_train, y_val


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning and k-fold evaluation for CNN (closed world)")
    parser.add_argument("--main_folder", type=str, required=True, help="Folder containing CSV files")
    parser.add_argument("--n", type=int, default=100, help="Number of send/receives per flow")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--k", type=int, default=10, help="Number of folds for k-fold cross-validation")
    parser.add_argument("--output_plot", type=str, default="accuracy_comparison.png", help="Filename to save the accuracy plot")
    args = parser.parse_args()

    m_values = [10, 25, 45]
    accuracies = {"With Features": [], "Without Features": []}

    for m in m_values:
        for include_features in [True, False]:
            print(f"Processing m={m}, include_features={include_features}")

            flows, labels, device_to_class = load_data(args.main_folder, m, args.n, include_features)
            print("Data loaded")

            X_train, X_val, y_train, y_val = train_val_split(flows, labels, args.val_split)

            print("Starting hyperparameter tuning")
            best_params = tune_hyperparameters(X_train, y_train, X_val, y_val, args.epochs)

            y_true_all, y_pred_all = train_and_evaluate_kfold(
                flows, labels, best_params, args.k, args.epochs, args.val_split
            )

            acc = accuracy_score(y_true_all, y_pred_all)
            if include_features:
                accuracies["With Features"].append(acc)
            else:
                accuracies["Without Features"].append(acc)
            print(f"Accuracy for m={m}, include_features={include_features}: {acc:.4f}")

    plt.figure(figsize=(10, 6))
    width = 0.45
    x = range(len(m_values))

    plt.bar(x, accuracies["With Features"], label="With Features", width=width ,color='blue', align='center')
    plt.bar(x, accuracies["Without Features"], label="Without Features", width=width, color='orange', align='edge')

    plt.xticks(x, [f"m={m}" for m in m_values])
    plt.title("Comparison of Accuracies for Different m Values")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    plt.savefig(args.output_plot)
    print(f"Accuracy comparison plot saved as {args.output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
