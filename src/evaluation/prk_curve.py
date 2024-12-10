import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_prk_curve(model, X_test, y_test):
    """
    Plots the PR-K curve (Precision and Recall vs. Threshold).

    Parameters:
    - model: Trained model with `predict_proba` method.
    - X_test: Test features.
    - y_test: Test labels.

    Returns:
    - None
    """
    # Get probabilities for the positive class
    probabilities = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)

    # Plot Precision and Recall vs. Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('PR-K (Precision-Recall vs. Threshold)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    from naive_bayes_tuning import tune_naive_bayes
    _, tuned_nb = tune_naive_bayes("data/data_imp_rural.csv")

    from train_validation_test_split import train_validation_test_split
    _, _, X_test, _, _, y_test, _ = train_validation_test_split("data/data_imp_rural.csv")

    plot_prk_curve(tuned_nb, X_test, y_test)
