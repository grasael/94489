import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from train_validation_test_split import train_validation_test_split
from rural_vs_nonrural_split import rural_vs_nonrural_split
from src.models.xgboost import train_xgboost
from src.utils.train_validation_test_split import train_validation_test_split

def tune_xgboost(data_file):
    """
    Perform hyperparameter tuning for XGBoost and evaluate the tuned model.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Prints:
    - Best parameters and cross-validation score.
    - Validation and test set results (overall, magnet schools, non-magnet schools).
    """
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test, _ = train_validation_test_split(data_file)
    splits = rural_vs_nonrural_split(data_file)

    X_test_magnet_1 = splits.get('X_test_magnet_1', None)
    X_test_magnet_0 = splits.get('X_test_magnet_0', None)
    y_test_magnet_1 = splits.get('y_test_magnet_1', None)
    y_test_magnet_0 = splits.get('y_test_magnet_0', None)

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0]
    }

    # Initialize GridSearchCV
    xgb = XGBClassifier(eval_metric='logloss', scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]))
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='recall', cv=3, verbose=1)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Best parameters and score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Score: {grid_search.best_score_}")

    # Evaluate the best model on validation set
    best_xgb_model = grid_search.best_estimator_
    xgb_y_val = best_xgb_model.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, xgb_y_val, digits=4))
    print(f"Validation Recall: {recall_score(y_val, xgb_y_val):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, xgb_y_val, labels=[1, 0]))

    # Predict on test set
    xgb_y_test = best_xgb_model.predict(X_test)
    print("\nTest Results:")
    print(classification_report(y_test, xgb_y_test, digits=4))
    print(f"Test Recall: {recall_score(y_test, xgb_y_test):.5f}")
    print(f"Test Accuracy: {accuracy_score(y_test, xgb_y_test):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, xgb_y_test, labels=[1, 0]))

    # Tuned XGBoost with manual parameters
    tuned_xgb = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        learning_rate=0.01,
        max_depth=4,
        n_estimators=50,
        subsample=0.6
    )
    tuned_xgb.fit(X_train, y_train)

    # Evaluate manually tuned model on validation set
    tuned_xgb_y_val = tuned_xgb.predict(X_val)
    print("\nTuned XGBoost Validation Results:")
    print(classification_report(y_val, tuned_xgb_y_val, digits=4))
    print(f"Validation Accuracy: {accuracy_score(y_val, tuned_xgb_y_val):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, tuned_xgb_y_val, labels=[1, 0]))

    # Evaluate manually tuned model on test set
    tuned_xgb_y_test = tuned_xgb.predict(X_test)
    print("\nTuned XGBoost Test Results:")
    print(classification_report(y_test, tuned_xgb_y_test, digits=4))
    print(f"Test Accuracy: {accuracy_score(y_test, tuned_xgb_y_test):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, tuned_xgb_y_test, labels=[1, 0]))

    # Evaluate on magnet school subsets (if available)
    if X_test_magnet_1 is not None and X_test_magnet_0 is not None:
        tuned_xgb_y_mag_test = tuned_xgb.predict(X_test_magnet_1)
        tuned_xgb_y_nomag_test = tuned_xgb.predict(X_test_magnet_0)
        print("\nMagnet Schools Tuned XGBoost Test Set Results:")
        print(classification_report(y_test_magnet_1, tuned_xgb_y_mag_test, digits=4))
        print("\nNon-Magnet Schools Tuned XGBoost Test Set Results:")
        print(classification_report(y_test_magnet_0, tuned_xgb_y_nomag_test, digits=4))

    return best_xgb_model, tuned_xgb

if __name__ == "__main__":
    tune_xgboost("data/data_imp_rural.csv")
