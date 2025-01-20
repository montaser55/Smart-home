import os
import matplotlib.pyplot as plt


def extract_data(input_dir = "../output/taskde"):

    m90_data = {"SVM": [], "k-NN": [], "Random Forest": [], "ensemble": []}
    m200_data = {"SVM": [], "k-NN": [], "Random Forest": [], "ensemble": []}
    n_values = []

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)

            parts = filename.split("_")
            n_value = parts[1]
            m_value = parts[2].split(".")[0]

            with open(filepath, "r") as file:
                lines = file.readlines()

            data = {"SVM": None, "k-NN": None, "Random Forest": None, "ensemble": None}
            for line in lines:
                if "Classifier: SVM" in line:
                    data["SVM"] = float(lines[lines.index(line) + 1].split(":")[1].strip())
                elif "Classifier: k-NN" in line:
                    data["k-NN"] = float(lines[lines.index(line) + 1].split(":")[1].strip())
                elif "Classifier: Random Forest" in line:
                    data["Random Forest"] = float(lines[lines.index(line) + 1].split(":")[1].strip())
                elif "Classifier: ensemble" in line:
                    data["ensemble"] = float(lines[lines.index(line) + 1].split(":")[1].strip())

            if m_value == "m90":
                n_values.append(n_value[1:])
                for clf in data:
                    m90_data[clf].append(data[clf])
            elif m_value == "m200":
                for clf in data:
                    m200_data[clf].append(data[clf])

    return m90_data, m200_data, n_values


def plot_data(m90_data, m200_data, n_values, output_dir="../output/taskde"):
    plt.figure(figsize=(10, 6))
    x = range(len(n_values))
    for clf, accuracies in m90_data.items():
        plt.plot(x, accuracies, marker='o', label=clf)
    plt.xticks(x, n_values)
    plt.title("Accuracy for m=90")
    plt.xlabel("n values")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    output_file = os.path.join(output_dir, "m90_accuracy_plot.png")
    plt.savefig(output_file)
    plt.close()

    plt.figure(figsize=(10, 6))
    for clf, accuracies in m200_data.items():
        plt.plot(x, accuracies, marker='o', label=clf)
    plt.xticks(x, n_values)
    plt.title("Accuracy for m=200")
    plt.xlabel("n values")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    output_file = os.path.join(output_dir, "m200_accuracy_plot.png")
    plt.savefig(output_file)
    plt.close()

def main():
    m90data, m200data, n_values = extract_data()
    plot_data(m90data, m200data, n_values)

if __name__ == "__main__":
    main()