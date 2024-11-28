import argparse
import random
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Machine Learning Classifier with Ensemble and Scaling")
    parser.add_argument("--scenario", choices=["open", "closed"], required=True,
                        help="Evaluation scenario: 'open' for open-world or 'closed' for closed-world.")
    parser.add_argument("--k_folds", type=int, required=True,
                        help="Number of folds for k-fold cross-validation.")
    parser.add_argument("--ensemble_method", choices=["random", "highest_confidence", "p1_p2_diff"], required=True,
                        help="Ensemble method: 'random', 'highest_confidence', or 'p1_p2_diff'.")
    parser.add_argument("--scaling_method", choices=["min_max", "z_score"], required=True,
                        help="Scaling method: 'min_max' or 'z_score'")
    return parser.parse_args()


def generate_dataset(n_samples=1000, n_features=5, n_classes=3):
    """
    Simulate a dataset for testing purposes.
    """
    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y


def k_fold_split(X, y, k_folds):
    """
    Split the dataset into k folds for cross-validation.
    """
    indices = np.arange(len(X))
    random.shuffle(indices)
    fold_size = len(X) // k_folds
    folds = []
    for i in range(k_folds):
        test_idx = indices[i * fold_size: (i + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def closed_world_scenario(X, y):
    """
    Closed-world scenario: All classes and communication pairs are known.
    """
    return X, y


def open_world_scenario(X, y, known_percentage=0.7):
    """
    Open-world scenario: Only part of the classes are known.
    """
    known_classes = np.unique(y)[:int(len(np.unique(y)) * known_percentage)]
    mask_known = np.isin(y, known_classes)
    X_known, y_known = X[mask_known], y[mask_known]
    X_unknown, y_unknown = X[~mask_known], y[~mask_known]
    return X_known, y_known, X_unknown, y_unknown


def normalize_dataset(X, method):
    """
    Normalize the dataset using the specified scaling method.
    """
    if method == "min_max":
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        return (X - X_min) / (X_max - X_min)
    elif method == "z_score":
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        return (X - X_mean) / X_std
    else:
        raise ValueError("Unsupported scaling method. Use 'min_max' or 'z_score'.")


def train_test_models(X_train, y_train, X_test, y_test, ensemble_method):
    """
    Train and test SVM, k-NN, Random Forest, and Ensemble Classifiers.
    """
    classifiers = {
        "SVM": SVC(probability=True),
        "k-NN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    predictions = {}
    confidences = {}

    # Train and test individual classifiers
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        predictions[name] = clf.predict(X_test)
        confidences[name] = clf.predict_proba(X_test)

    # Ensemble classification
    ensemble_predictions = []
    for i in range(len(X_test)):
        if ensemble_method == "random":
            chosen_classifier = random.choice(list(classifiers.keys()))
            ensemble_predictions.append(predictions[chosen_classifier][i])
        elif ensemble_method == "highest_confidence":
            highest_confidence_clf = max(confidences, key=lambda clf: np.max(confidences[clf][i]))
            ensemble_predictions.append(predictions[highest_confidence_clf][i])
        elif ensemble_method == "p1_p2_diff":
            max_diff = -1
            chosen_class = None
            for clf, prob in confidences.items():
                sorted_prob = np.sort(prob[i])[::-1]
                diff = sorted_prob[0] - sorted_prob[1]
                if diff > max_diff:
                    max_diff = diff
                    chosen_class = predictions[clf][i]
            ensemble_predictions.append(chosen_class)

    # Report results
    print("\n=== Individual Classifier Reports ===")
    for name in classifiers:
        print(f"Classifier: {name}")
        print(classification_report(y_test, predictions[name]))

    print("\n=== Ensemble Classifier Report ===")
    print(classification_report(y_test, ensemble_predictions))


def main():
    args = parse_args()

    # Generate synthetic dataset
    X, y = generate_dataset()

    # Normalize dataset
    X = normalize_dataset(X, args.scaling_method)

    # Scenario-based evaluation
    if args.scenario == "closed":
        X, y = closed_world_scenario(X, y)
        k_folds = k_fold_split(X, y, args.k_folds)
        for train_idx, test_idx in k_folds:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_test_models(X_train, y_train, X_test, y_test, args.ensemble_method)

    elif args.scenario == "open":
        X_known, y_known, X_unknown, y_unknown = open_world_scenario(X, y)
        k_folds = k_fold_split(X_known, y_known, args.k_folds)
        for train_idx, test_idx in k_folds:
            X_train, X_test = X_known[train_idx], X_known[test_idx]
            y_train, y_test = y_known[train_idx], y_known[test_idx]
            train_test_models(X_train, y_train, X_test, y_test, args.ensemble_method)


if __name__ == "__main__":
    main()
