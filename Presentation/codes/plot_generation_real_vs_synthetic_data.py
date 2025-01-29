import numpy as np
import matplotlib.pyplot as plt

# Data
data_types = ["Real Data", "Synthetic Data"]
classifiers = ["SVM", "k-NN", "Random Forest", "Ensemble"]
accuracy_values = [
    [0.56, 0.51],  # SVM
    [0.62, 0.49],  # k-NN
    [0.87, 0.71],  # Random Forest
    [0.76, 0.61]   # Ensemble
]

# Bar plot settings
x = np.arange(len(classifiers))  # Classifier indices
width = 0.3  # Adjusted width for two bars per group

# Colors for different data types
colors = ['#2ca02c', '#d62728']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, data_type in enumerate(data_types):
    ax.bar(x + i * width - width / 2, [acc[i] for acc in accuracy_values], width, label=data_type, color=colors[i])

# Labels & Titles
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.set_xlabel("Classifiers")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison: Real Data vs Synthetic Data")
ax.legend(title="Data Type")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Save and close plot
plt.savefig("../output/compare_real_vs_synthetic.png", bbox_inches="tight")
plt.close()
