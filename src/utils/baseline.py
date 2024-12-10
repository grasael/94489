import numpy as np
from sklearn.metrics import precision_score, recall_score
from rural_vs_nonrural_split import rural_vs_nonrural_split

def compute_baseline(data_file):
    """
    Compute baseline performance metrics for overall, rural, and non-rural subsets.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Prints:
    - Precision and recall metrics for overall, rural, and non-rural datasets.
    """
    splits = rural_vs_nonrural_split(data_file)

    # Overall baseline
    y_test = splits['y_test']
    X_test_sorted = splits['X_test']
    X_test_sorted['title_word_count'] = X_test_sorted['title'].str.split().str.len()
    X_test_sorted = X_test_sorted.sort_values(by='title_word_count', ascending=True)

    threshold = int(len(X_test_sorted) * 0.306)
    baseline_predictions = np.zeros(len(X_test_sorted))
    baseline_predictions[:threshold] = 1

    baseline_recall = recall_score(y_test, baseline_predictions)
    baseline_precision = precision_score(y_test, baseline_predictions)
    print(f"Overall Baseline Recall: {baseline_recall}")
    print(f"Overall Baseline Precision: {baseline_precision}")

    # Rural metrics
    y_test_rural = splits['y_test_rural']
    rural_baseline = baseline_predictions[:len(y_test_rural)]
    rural_recall = recall_score(y_test_rural, rural_baseline)
    rural_precision = precision_score(y_test_rural, rural_baseline)
    print(f"Rural Baseline Recall: {rural_recall}")
    print(f"Rural Baseline Precision: {rural_precision}")

    # Non-Rural metrics
    y_test_nonrural = splits['y_test_nonrural']
    nonrural_baseline = baseline_predictions[len(y_test_rural):]
    nonrural_recall = recall_score(y_test_nonrural, nonrural_baseline)
    nonrural_precision = precision_score(y_test_nonrural, nonrural_baseline)
    print(f"Non-Rural Baseline Recall: {nonrural_recall}")
    print(f"Non-Rural Baseline Precision: {nonrural_precision}")

if __name__ == "__main__":
    compute_baseline("data/data_imp_rural.csv")
