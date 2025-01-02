import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from memory_profiler import memory_usage
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def main():
    # Define classifiers
    classifiers = {
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'k-NN': KNeighborsClassifier()
    }

    # Placeholder for runtime, memory, and error data
    results = {
        'SVM': {'errors': [], 'runtime': {'train': [], 'test': []}, 'memory': {'train': [], 'test': []}},
        'Random Forest': {'errors': [], 'runtime': {'train': [], 'test': []}, 'memory': {'train': [], 'test': []}},
        'k-NN': {'errors': [], 'runtime': {'train': [], 'test': []}, 'memory': {'train': [], 'test': []}}
    }

    # Use your dataset here (X_train, y_train, X_test, y_test)
    # Replace the dummy data below with your actual train/test sets
    X_train, X_test = np.random.rand(100, 20), np.random.rand(30, 20)
    y_train, y_test = np.random.randint(0, 3, 100), np.random.randint(0, 3, 30)

    # Analyze each classifier
    for clf_name, clf in classifiers.items():
        # Measure runtime and memory during training
        start_time = time.time()
        mem_usage_train = memory_usage((clf.fit, (X_train, y_train)))
        train_time = time.time() - start_time

        # Measure runtime and memory during testing
        start_time = time.time()
        mem_usage_test = memory_usage((clf.predict, (X_test,)))
        test_time = time.time() - start_time

        # Record runtime and memory usage
        results[clf_name]['runtime']['train'].append(train_time)
        results[clf_name]['runtime']['test'].append(test_time)
        results[clf_name]['memory']['train'].append(max(mem_usage_train) - min(mem_usage_train))
        results[clf_name]['memory']['test'].append(max(mem_usage_test) - min(mem_usage_test))

        # Predict and record errors
        y_pred = clf.predict(X_test)
        errors = np.where(y_pred != y_test)[0]
        results[clf_name]['errors'].extend(errors)

    # Step 2: Generate Venn Diagrams for Errors
    svm_errors = set(results['SVM']['errors'])
    rf_errors = set(results['Random Forest']['errors'])
    knn_errors = set(results['k-NN']['errors'])

    plt.figure(figsize=(10, 5))
    venn3([svm_errors, rf_errors, knn_errors], ('SVM', 'Random Forest', 'k-NN'))
    plt.title("Venn Diagram of Classification Errors")
    plt.show()

    # Step 3: Runtime and Memory Plots
    # Runtime
    train_times = [np.mean(results[clf]['runtime']['train']) for clf in classifiers]
    test_times = [np.mean(results[clf]['runtime']['test']) for clf in classifiers]

    plt.figure(figsize=(10, 5))
    plt.bar(classifiers.keys(), train_times, alpha=0.7, label='Training Time')
    plt.bar(classifiers.keys(), test_times, alpha=0.7, label='Testing Time')
    plt.title("Runtime Analysis")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.show()

    # Memory
    train_memory = [np.mean(results[clf]['memory']['train']) for clf in classifiers]
    test_memory = [np.mean(results[clf]['memory']['test']) for clf in classifiers]

    plt.figure(figsize=(10, 5))
    plt.bar(classifiers.keys(), train_memory, alpha=0.7, label='Training Memory')
    plt.bar(classifiers.keys(), test_memory, alpha=0.7, label='Testing Memory')
    plt.title("Memory Usage Analysis")
    plt.ylabel("Memory Usage (MB)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
