import numpy as np
import matplotlib.pyplot as plt

# Data
k_values = ["k=5", "k=10"]
classifiers = ["SVM", "k-NN", "Random Forest", "Ensemble"]
accuracy_values = [
    [0.56, 0.54],  # SVM
    [0.62, 0.54],  # k-NN
    [0.87, 0.88],  # Random Forest
    [0.76, 0.74]   # Ensemble
]

# Bar plot settings
x = np.arange(len(classifiers))  # Classifier indices
width = 0.3  # Adjusted width for two bars per group

# Colors for different k values
colors = ['#9467bd', '#8c564b']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, k in enumerate(k_values):
    ax.bar(x + i * width - width / 2, [acc[i] for acc in accuracy_values], width, label=k, color=colors[i])

# Labels & Titles
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.set_xlabel("Classifiers")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison Across Different k Values in K-Fold Cross-Validation")
ax.legend(title="k Values")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save and close plot
plt.savefig("../output/compare_k_values.png")
plt.close()
