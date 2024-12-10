import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from train_validation_test_split import train_validation_test_split  # Preprocessing split
from rural_vs_nonrural_split import rural_vs_nonrural_split  # Rural/Non-Rural split
from src.utils.train_validation_test_split import train_validation_test_split
from src.evaluation.feature_importance import plot_feature_importance

def catboost_model(data_file):
    """
    Train and evaluate a CatBoost model.

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

    # Train CatBoost Classifier
    catboost_model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function='Logloss',
        cat_features=[]  # Add the indices of categorical features here if needed
    )

    catboost_model.fit(X_train, y_train)

    # Evaluate on validation set
    cb_y_val = catboost_model.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, cb_y_val, digits=4))

    # Evaluate on rural validation set
    cb_y_rural_val = catboost_model.predict(X_val_rural_1)
    print("\nRural Validation Results:")
    print(classification_report(y_val_rural_1, cb_y_rural_val, digits=4))

    # Evaluate on non-rural validation set
    cb_y_norural_val = catboost_model.predict(X_val_rural_0)
    print("\nNon-Rural Validation Results:")
    print(classification_report(y_val_rural_0, cb_y_norural_val, digits=4))

    return catboost_model

if __name__ == "__main__":
    # Example usage
    catboost_model("data/data_imp_rural.csv")
