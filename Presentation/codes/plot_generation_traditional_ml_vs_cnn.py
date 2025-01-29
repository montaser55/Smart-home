import numpy as np
import matplotlib.pyplot as plt

# Data
classifiers = ["SVM", "k-NN", "Random Forest", "Ensemble", "CNN", "CNN without features"]
accuracy_values = [0.56, 0.62, 0.87, 0.76, 0.83, 0.87]  # Accuracy values for each classifier

# Bar plot settings
x = np.arange(len(classifiers))  # Classifier indices
width = 0.5  # Width of bars

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, accuracy_values, width, color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'blue', 'orange'])

# Annotate bars with accuracy value
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom')

# Labels & Titles
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.set_xlabel("Classifiers")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy Comparison for Different Classifiers")
ax.grid(axis="y", linestyle="--", alpha=0.7)

# Display plot
plt.savefig("../output/compare_classifiers.png")
plt.close()
