import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from train_validation_test_split import train_validation_test_split  # Preprocessing split
from rural_vs_nonrural_split import rural_vs_nonrural_split  # Rural/Non-Rural split
from feature_mapping import apply_feature_name_mapping  # Feature name mapping
from src.utils.train_validation_test_split import train_validation_test_split
from src.evaluation.feature_importance import plot_feature_importance

def gradient_boosting_model(data_file):
    """
    Train and evaluate a Gradient Boosting model.

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

    # Train Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(random_state=0)
    gbc.fit(X_train, y_train)

    # Evaluate on validation set
    gbc_y_val = gbc.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, gbc_y_val, digits=4))
    print(f"Validation Accuracy: {accuracy_score(y_val, gbc_y_val):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, gbc_y_val, labels=[1, 0]))

    # Evaluate on rural validation set
    gbc_y_rural_val = gbc.predict(X_val_rural_1)
    print("\nRural Validation Results:")
    print(classification_report(y_val_rural_1, gbc_y_rural_val, digits=4))

    # Evaluate on non-rural validation set
    gbc_y_norural_val = gbc.predict(X_val_rural_0)
    print("\nNon-Rural Validation Results:")
    print(classification_report(y_val_rural_0, gbc_y_norural_val, digits=4))

    # Feature importance
    gbc_importances = gbc.feature_importances_
    feature_names = ct.get_feature_names_out()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': gbc_importances})
    importance_df = apply_feature_name_mapping(importance_df)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Filter and display top 10 features
    importance_df_clean = importance_df.head(10)
    print("\nTop 10 Features:")
    print(importance_df_clean)

    # Plot feature importances
    plt.figure(figsize=(12, 15))
    plt.barh(importance_df_clean['Feature'], importance_df_clean['Importance'], align='center')
    plt.xlabel('Importance')
    plt.title('Feature Importance from Gradient Boosting')
    plt.gca().invert_yaxis()
    plt.show()

    return gbc

if __name__ == "__main__":
    gradient_boosting_model("data/data_imp_rural.csv")
