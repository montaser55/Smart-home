import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def plot_venn_diagram(true_labels, predicted_labels, title, same_wrong_label=False):
    errors = {
        model: set(i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred)
        for model, predictions in predicted_labels.items()
    }

    svm_errors = errors.get("SVM", set())
    rf_errors = errors.get("Random Forest", set())
    knn_errors = errors.get("k-NN", set())

    if same_wrong_label:

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

        venn3(subsets=(len(only_svm), len(only_rf), len(svm_rf), len(only_knn), len(knn_svm), len(rf_knn), len(all_three)), set_labels=("SVM", "Random Forest", "k-NN"))
        plt.title(title)
    else:
        venn3((svm_errors, rf_errors, knn_errors), set_labels=("SVM", "Random Forest", "k-NN"))
        plt.title(title)
    plt.show()


def plot_runtime_memory(runtime_memory_logs):
    models = ["SVM", "Random Forest", "k-NN"]
    training_times = []
    testing_times = []
    training_memory = []
    testing_memory = []

    for model in models:
        train_logs = runtime_memory_logs[model][0]["train"]
        test_logs = runtime_memory_logs[model][0]["test"]

        training_times.append(train_logs[0]["runtime_seconds"])
        testing_times.append(test_logs[0]["runtime_seconds"])
        training_memory.append(train_logs[0]["memory_peak_kb"])
        testing_memory.append(test_logs[0]["memory_peak_kb"])

    x = range(len(models))

    # Plot runtime
    plt.bar(x, training_times, width=0.4, label="Training Time", align='center')
    plt.bar(x, testing_times, width=0.4, label="Testing Time", align='edge')
    plt.xticks(x, models)
    plt.xlabel("Classifiers")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison")
    plt.legend()
    plt.show()

    # Plot memory usage
    plt.bar(x, training_memory, width=0.4, label="Training Memory", align='center')
    plt.bar(x, testing_memory, width=0.4, label="Testing Memory", align='edge')
    plt.xticks(x, models)
    plt.xlabel("Classifiers")
    plt.ylabel("Memory Usage (KB)")
    plt.title("Memory Usage Comparison")
    plt.legend()
    plt.show()


def main():
    json_file = "../task1/output/1c_n10_data.json"
    with open(json_file, 'r') as file:
        data = json.load(file)

    true_labels = data["true_labels"]
    predicted_labels = data["predicted_labels"]
    runtime_memory_logs = data["runtime_memory_logs"]

    print(f"Length of labels: {len(true_labels)}")

    plot_venn_diagram(true_labels, predicted_labels, "Classification Errors")
    plot_venn_diagram(true_labels, predicted_labels, "Classification Errors with Same Wrong Label", same_wrong_label=True)
    plot_runtime_memory(runtime_memory_logs)


if __name__ == "__main__":
    main()
