import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import argparse
import os


def plot_venn_diagrams(true_labels, predicted_labels, output_dir, input_file):
    errors = {
        model: set(i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred)
        for model, predictions in predicted_labels.items()
    }

    svm_errors = errors.get("SVM", set())
    rf_errors = errors.get("Random Forest", set())
    knn_errors = errors.get("k-NN", set())

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    plt.sca(axs[0])
    venn3((svm_errors, rf_errors, knn_errors), set_labels=("SVM", "Random Forest", "k-NN"))
    plt.title("Classification Errors")

    only_svm = svm_errors - rf_errors - knn_errors
    only_rf = rf_errors - svm_errors - knn_errors
    only_knn = knn_errors - svm_errors - rf_errors

    svm_rf = set(
        idx for idx in svm_errors & rf_errors
        if predicted_labels["SVM"][idx] == predicted_labels["Random Forest"][idx]
        if idx not in knn_errors or predicted_labels["k-NN"][idx] != predicted_labels["SVM"][idx]
    )

    rf_knn = set(
        idx for idx in rf_errors & knn_errors
        if predicted_labels["Random Forest"][idx] == predicted_labels["k-NN"][idx]
        if idx not in svm_errors or predicted_labels["SVM"][idx] != predicted_labels["Random Forest"][idx]
    )

    knn_svm = set(
        idx for idx in knn_errors & svm_errors
        if predicted_labels["k-NN"][idx] == predicted_labels["SVM"][idx]
        if idx not in rf_errors or predicted_labels["Random Forest"][idx] != predicted_labels["k-NN"][idx]
    )

    all_three = set(
        idx for idx in svm_errors & rf_errors & knn_errors
        if predicted_labels["SVM"][idx] == predicted_labels["Random Forest"][idx] == predicted_labels["k-NN"][idx]
    )

    plt.sca(axs[1])
    venn3(subsets=(len(only_svm), len(only_rf), len(svm_rf), len(only_knn), len(knn_svm), len(rf_knn), len(all_three)), set_labels=("SVM", "Random Forest", "k-NN"))
    plt.title("Classification Errors (Same Wrong Label)")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}_venn.png")
    plt.savefig(output_path)
    plt.close()


def plot_runtime_and_memory(runtime_memory_logs, output_dir, input_file):
    models = ["SVM", "Random Forest", "k-NN"]
    training_times = []
    testing_times = []
    training_memory = []
    testing_memory = []

    for model in models:
        total_training_time = 0
        total_testing_time = 0
        total_training_memory = 0
        total_testing_memory = 0
        total_logs = 0

        for e in runtime_memory_logs[model]:
            train_logs = e["train"]
            test_logs = e["test"]

            total_training_time += sum(log["runtime_seconds"] for log in train_logs)
            total_testing_time += sum(log["runtime_seconds"] for log in test_logs)
            total_training_memory += sum(log["memory_peak_kb"] for log in train_logs)
            total_testing_memory += sum(log["memory_peak_kb"] for log in test_logs)
            total_logs += len(train_logs)

        training_times.append(total_training_time / total_logs)
        testing_times.append(total_testing_time / total_logs)
        training_memory.append(total_training_memory / total_logs)
        testing_memory.append(total_testing_memory / total_logs)

    x = range(len(models))

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].bar(x, training_times, width=0.4, label="Training Time", align='center')
    axs[0].bar(x, testing_times, width=0.4, label="Testing Time", align='edge')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(models)
    axs[0].set_xlabel("Classifiers")
    axs[0].set_ylabel("Runtime (seconds)")
    axs[0].set_yscale("log")
    axs[0].set_title("Runtime Comparison")
    axs[0].legend()

    axs[1].bar(x, training_memory, width=0.4, label="Training Memory", align='center')
    axs[1].bar(x, testing_memory, width=0.4, label="Testing Memory", align='edge')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(models)
    axs[1].set_xlabel("Classifiers")
    axs[1].set_ylabel("Memory Usage (KB)")
    axs[1].set_title("Memory Usage Comparison")
    axs[1].legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{os.path.splitext(input_file)[0]}_runtime_memory.png")
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize classifier performance")
    parser.add_argument("--input_dir", type=str, help="Directory of the JSON file")
    parser.add_argument("--input_file", type=str, help="Name of the JSON file")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output plots")

    args = parser.parse_args()

    json_file = os.path.join(args.input_dir, args.input_file)

    with open(json_file, 'r') as file:
        data = json.load(file)

    true_labels = data["true_labels"]
    predicted_labels = data["predicted_labels"]
    runtime_memory_logs = data["runtime_memory_logs"]

    print(f"Length of labels: {len(true_labels)}")

    plot_venn_diagrams(true_labels, predicted_labels, args.output_dir, args.input_file)
    plot_runtime_and_memory(runtime_memory_logs, args.output_dir, args.input_file)


if __name__ == "__main__":
    main()
