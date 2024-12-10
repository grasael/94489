from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score, classification_report, confusion_matrix, accuracy_score
from train_validation_test_split import train_validation_test_split
from rural_vs_nonrural_split import rural_vs_nonrural_split
from src.models.xgboost import train_xgboost
from src.utils.train_validation_test_split import train_validation_test_split

def tune_naive_bayes(data_file):
    """
    Perform hyperparameter tuning for Naive Bayes and evaluate the tuned model.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Prints:
    - Best parameters and cross-validation score.
    - Validation, test, and rural/non-rural test set results.
    """
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test, _ = train_validation_test_split(data_file)
    splits = rural_vs_nonrural_split(data_file)

    X_test_rural_1 = splits['X_test_rural_1']
    X_test_rural_0 = splits['X_test_rural_0']
    y_test_rural_1 = splits['y_test_rural_1']
    y_test_rural_0 = splits['y_test_rural_0']

    # Define parameter grid
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 5.0],
        'fit_prior': [True, False]
    }

    # Initialize GridSearchCV
    nb = MultinomialNB()
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, scoring='recall', cv=5, verbose=1, n_jobs=-1)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Score: {grid_search.best_score_}")

    # Evaluate the best model on validation set
    best_nb_model = grid_search.best_estimator_
    nb_y_val = best_nb_model.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, nb_y_val, digits=4))
    print(f"Validation Recall: {recall_score(y_val, nb_y_val):.5f}")
    print(f"Validation Accuracy: {accuracy_score(y_val, nb_y_val):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, nb_y_val, labels=[1, 0]))

    # Evaluate on test set
    nb_y_test = best_nb_model.predict(X_test)
    print("\nTest Results:")
    print(classification_report(y_test, nb_y_test, digits=4))
    print(f"Test Recall: {recall_score(y_test, nb_y_test):.5f}")
    print(f"Test Accuracy: {accuracy_score(y_test, nb_y_test):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, nb_y_test, labels=[1, 0]))

    # Tuned Naive Bayes with manual parameters
    tuned_nb = MultinomialNB(alpha=0.5, fit_prior=False)
    tuned_nb.fit(X_train, y_train)

    # Evaluate manually tuned model on validation set
    tuned_nb_y_val = tuned_nb.predict(X_val)
    print("\nTuned Naive Bayes Validation Results:")
    print(classification_report(y_val, tuned_nb_y_val, digits=4))
    print(f"Validation Accuracy: {accuracy_score(y_val, tuned_nb_y_val):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, tuned_nb_y_val, labels=[1, 0]))

    # Evaluate manually tuned model on test set
    tuned_nb_y_test = tuned_nb.predict(X_test)
    print("\nTuned Naive Bayes Test Results:")
    print(classification_report(y_test, tuned_nb_y_test, digits=4))
    print(f"Test Accuracy: {accuracy_score(y_test, tuned_nb_y_test):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, tuned_nb_y_test, labels=[1, 0]))

    # Evaluate manually tuned model on rural/non-rural test sets
    tuned_nb_y_rural_test = tuned_nb.predict(X_test_rural_1)
    tuned_nb_y_norural_test = tuned_nb.predict(X_test_rural_0)
    print("\nRural Schools Tuned Naive Bayes Test Results:")
    print(classification_report(y_test_rural_1, tuned_nb_y_rural_test, digits=4))
    print("\nNon-Rural Schools Tuned Naive Bayes Test Results:")
    print(classification_report(y_test_rural_0, tuned_nb_y_norural_test, digits=4))

    return best_nb_model, tuned_nb

if __name__ == "__main__":
    tune_naive_bayes("data/data_imp_rural.csv")
