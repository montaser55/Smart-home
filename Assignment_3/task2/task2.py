import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, Dropout, Dense, Flatten, Input, Activation, MaxPooling1D
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def process_flow(data, n, include_features=True):
    padded_data = np.pad(data, ((0, max(0, n - len(data))), (0, 0)), mode='constant')

    if include_features:
        total_packets = len(data)
        incoming_packets = np.sum(data[:, 0] < 0)
        outgoing_packets = total_packets - incoming_packets
        incoming_ratio = incoming_packets / total_packets if total_packets > 0 else 0
        outgoing_ratio = outgoing_packets / total_packets if total_packets > 0 else 0
        cumulative_features = [total_packets, incoming_packets, outgoing_packets, incoming_ratio, outgoing_ratio]
        padded_data = np.hstack((padded_data, np.tile(cumulative_features, (n, 1))))

    return padded_data


def preprocess_csv_files(main_folder):
    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        if "Packet Size" not in df.columns or "Direction" not in df.columns:
                            raise ValueError(f"CSV {file_path} is missing required columns.")

                        df["Packet Size"] = df.apply(lambda row: -row["Packet Size"] if row["Direction"] == "A to B" else row["Packet Size"], axis=1)
                        df = df.drop(columns=["Direction"])
                        df.to_csv(file_path, index=False)
                        print(f"Processed file: {file_path}")
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")


def load_data(main_folder, m, n, include_features=True):
    flows = []
    labels = []
    folder_to_class = {}
    class_idx = 0

    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)
        if os.path.isdir(folder_path):
            folder_to_class[folder] = class_idx
            class_idx += 1

            files = []
            for f in os.listdir(folder_path):
                if f.endswith('.csv'):
                    files.append(f)

            num_flows = min(m, len(files))
            if m >= 48:
                print(f"Taking 48 flows from {folder} because it is the max.")

            for file in files[:num_flows]:
                flow_path = os.path.join(folder_path, file)
                data = pd.read_csv(flow_path, header=0).values[:n]
                padded_data = process_flow(data, n, include_features)

                flows.append(padded_data)
                labels.append(folder_to_class[folder])

    flows = np.array(flows, dtype=np.float32)
    labels = tf.keras.utils.to_categorical(labels, num_classes=class_idx).astype(np.float32)
    return flows, labels, folder_to_class


def classify_new_flow(model, flow_path, n, include_features=True):
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"The file {flow_path} does not exist. Please check the path.")

    data = pd.read_csv(flow_path, header=0)
    try:
        data = data.astype(np.float32).values[:n]
    except ValueError as e:
        raise ValueError(f"Non-numeric or invalid data found in {flow_path}.") from e

    padded_data = process_flow(data, n, include_features)
    padded_data = np.array(padded_data, dtype=np.float32)
    padded_data = np.expand_dims(padded_data, axis=0)
    prediction = model.predict(padded_data)
    return np.argmax(prediction)


def build_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))

    for i in [64, 128, 256, 512]:
        model.add(Conv1D(i, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Conv1D(i, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

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


parser = argparse.ArgumentParser(description="Deep Learning Classifier for Flows")
parser.add_argument("--main_folder", type=str, required=True, help="Path to the main folder containing device folders.")
parser.add_argument("--m", type=int, default=48, help="Number of flows to take from each folder (max 48).")
parser.add_argument("--n", type=int, required=True, help="Number of lines (packet sizes) to take from each flow.")
parser.add_argument("--include_features", action="store_true", help="Include cumulative features (default: enabled).")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
parser.add_argument("--test_split", type=float, default=0.1, help="Proportion of test dataset.")
parser.add_argument("--val_split", type=float, default=0.1, help="Proportion of validation dataset.")
parser.add_argument("--new_flow", type=str, required=True, help="Path to a new flow file for classification.")
parser.add_argument("--preprocess", action="store_true", help="Run preprocessing on CSV files.")
args = parser.parse_args()

if args.preprocess:
    preprocess_csv_files(args.main_folder)

flows, labels, folder_to_class = load_data(args.main_folder, args.m, args.n, args.include_features)

train_flows, test_flows, train_labels, test_labels = train_test_split(flows, labels, test_size=args.test_split, stratify=np.argmax(labels, axis=1), random_state=42)

train_flows = np.tile(train_flows, (6, 1, 1))
train_labels = np.tile(train_labels, (6, 1))

input_shape = train_flows.shape[1:]
model = build_cnn(input_shape, len(folder_to_class))
model.fit(train_flows, train_labels, epochs=args.epochs, batch_size=args.batch_size, validation_split=args.val_split)

loss, accuracy = model.evaluate(test_flows, test_labels, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

y_prediction = model.predict(test_flows)
y_prediction_classes = np.argmax(y_prediction, axis=1)
y_true_classes = np.argmax(test_labels, axis=1)

report = classification_report(y_true_classes, y_prediction_classes, target_names=list(folder_to_class.keys()), digits=4)
print("\nClassification Report:")
print(report)

model.summary()

cm = confusion_matrix(y_true_classes, y_prediction_classes)
print(cm)

predicted_class = classify_new_flow(model, args.new_flow, args.n, args.include_features)
print(f"The new flow belongs to class: {predicted_class}")

print("\nFolder-to-Class Mapping:")
for folder, class_idx in folder_to_class.items():
    print(f"{folder}: Class {class_idx}")
