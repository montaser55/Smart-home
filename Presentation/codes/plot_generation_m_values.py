import numpy as np
import matplotlib.pyplot as plt

# Data
m_values = ["m=90", "m=150", "m=200"]
classifiers = ["SVM", "k-NN", "Random Forest", "Ensemble"]
accuracy_values = [
    [0.56, 0.56, 0.55],  # SVM
    [0.57, 0.62, 0.53],  # k-NN
    [0.86, 0.87, 0.85],  # Random Forest
    [0.79, 0.76, 0.77]   # Ensemble
]

# Bar plot settings
x = np.arange(len(classifiers))  # Classifier indices
width = 0.2  # Width of bars

# Colors for different m values
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, m in enumerate(m_values):
    ax.bar(x + i * width, [acc[i] for acc in accuracy_values], width, label=m, color=colors[i])

# Labels & Titles
ax.set_xticks(x + width)
ax.set_xticklabels(classifiers)
ax.set_xlabel("Classifiers")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison Across Different m Values")
ax.legend(title="m Values")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Display plot
plt.savefig("../output/compare_m_values.png")
plt.close()
