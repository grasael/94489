import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from train_validation_test_split import train_validation_test_split  # Preprocessing split
from rural_vs_nonrural_split import rural_vs_nonrural_split  # Rural/Non-Rural split
from feature_mapping import apply_feature_name_mapping  # Feature name mapping
from src.utils.train_validation_test_split import train_validation_test_split
from src.evaluation.feature_importance import plot_feature_importance

def xgboost_model(data_file):
    """
    Train and evaluate an XGBoost model.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Prints:
    - Classification reports for overall, rural, and non-rural data.
    - Top feature importances.
    """
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test, ct = train_validation_test_split(data_file)
    splits = rural_vs_nonrural_split(data_file)

    X_val_rural_1 = splits['X_val_rural_1']
    X_val_rural_0 = splits['X_val_rural_0']
    y_val_rural_1 = splits['y_val_rural_1']
    y_val_rural_0 = splits['y_val_rural_0']

    # Train XGBoost
    xgb = XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1])
    )
    xgb.fit(X_train, y_train)

    # Evaluate on validation set
    xgb_y_val = xgb.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, xgb_y_val, digits=4))
    print(f"Validation Accuracy: {accuracy_score(y_val, xgb_y_val):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, xgb_y_val, labels=[1, 0]))

    # Evaluate on test set
    xgb_y_test = xgb.predict(X_test)
    print("\nTest Results:")
    print(classification_report(y_test, xgb_y_test, digits=4))
    print(f"Test Accuracy: {accuracy_score(y_test, xgb_y_test):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, xgb_y_test, labels=[1, 0]))

    # Evaluate on rural validation set
    y_rural_val = xgb.predict(X_val_rural_1)
    print("\nRural Validation Results:")
    print(classification_report(y_val_rural_1, y_rural_val, digits=4))

    # Evaluate on non-rural validation set
    y_norural_val = xgb.predict(X_val_rural_0)
    print("\nNon-Rural Validation Results:")
    print(classification_report(y_val_rural_0, y_norural_val, digits=4))

    # Feature importance
    xgboost_importances = xgb.feature_importances_
    feature_names = ct.get_feature_names_out()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': xgboost_importances})
    importance_df = apply_feature_name_mapping(importance_df)
    importance_df_clean = importance_df[~importance_df['Feature'].str.contains("School State", na=False)].head(10)
    importance_df_clean = importance_df_clean.sort_values(by='Importance', ascending=True)

    # Plot feature importances
    plt.figure(figsize=(12, 15))
    plt.barh(importance_df_clean['Feature'], importance_df_clean['Importance'], align='center')
    plt.xlabel('Importance')
    plt.title('Feature Importance from XGBoost')
    plt.gca().invert_yaxis()
    plt.show()

    return xgb

if __name__ == "__main__":
    xgboost_model("data/data_imp_rural.csv")
