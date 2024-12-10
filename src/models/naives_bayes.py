import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from train_validation_test_split import train_validation_test_split  # Preprocessing split
from rural_vs_nonrural_split import rural_vs_nonrural_split  # Rural/Non-Rural split
from src.utils.train_validation_test_split import train_validation_test_split
from src.evaluation.feature_importance import plot_feature_importance

def naive_bayes_model(data_file):
    """
    Train and evaluate a Naive Bayes model.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Prints:
    - Classification reports for overall, rural, and non-rural data.
    """
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test, _ = train_validation_test_split(data_file)
    splits = rural_vs_nonrural_split(data_file)

    X_val_rural_1 = splits['X_val_rural_1']
    X_val_rural_0 = splits['X_val_rural_0']
    y_val_rural_1 = splits['y_val_rural_1']
    y_val_rural_0 = splits['y_val_rural_0']

    # Train Naive Bayes Classifier
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    # Evaluate on validation set
    nb_y_val = nb.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, nb_y_val, digits=4))
    print(f"Validation Accuracy: {accuracy_score(y_val, nb_y_val):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, nb_y_val, labels=[1, 0]))

    # Evaluate on rural validation set
    nb_y_rural_val = nb.predict(X_val_rural_1)
    print("\nRural Validation Results:")
    print(classification_report(y_val_rural_1, nb_y_rural_val, digits=4))

    # Evaluate on non-rural validation set
    nb_y_norural_val = nb.predict(X_val_rural_0)
    print("\nNon-Rural Validation Results:")
    print(classification_report(y_val_rural_0, nb_y_norural_val, digits=4))

    return nb

if __name__ == "__main__":
    naive_bayes_model("data/data_imp_rural.csv")
