import numpy as np
import matplotlib.pyplot as plt

# Data
n_values = ["n=10", "n=25", "n=45"]
classifiers = ["SVM", "k-NN", "Random Forest", "Ensemble"]
accuracy_values = [
    [0.55, 0.56, 0.49],  # SVM
    [0.48, 0.62, 0.48],  # k-NN
    [0.86, 0.87, 0.89],  # Random Forest
    [0.74, 0.76, 0.71]   # Ensemble
]

# Bar plot settings
x = np.arange(len(classifiers))  # Classifier indices
width = 0.2  # Width of bars

# Colors for different n values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, n in enumerate(n_values):
    ax.bar(x + i * width, [acc[i] for acc in accuracy_values], width, label=n, color=colors[i])

# Labels & Titles
ax.set_xticks(x + width)
ax.set_xticklabels(classifiers)
ax.set_xlabel("Classifiers")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison Across Different n Values")
ax.legend(title="n Values")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Display plot
plt.savefig("../output/compare_n_values.png")
plt.close()
