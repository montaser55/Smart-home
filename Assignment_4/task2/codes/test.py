from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example data
y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]  # Actual labels
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]  # Predicted labels

# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.title('Confusion Matrix')
plt.show()
