import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from src.utils.feature_mapping import apply_feature_name_mapping  # Import the function
from train_validation_test_split import train_validation_test_split  # Import your data preprocessing script
from rural_vs_nonrural_split import rural_vs_nonrural_split  # Import rural/non-rural split
from src.utils.train_validation_test_split import train_validation_test_split
from src.evaluation.feature_importance import plot_feature_importance

def logistic_regression(data_file):
    """
    Train and evaluate a Logistic Regression model.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Prints:
    - Classification reports and confusion matrices for overall, rural, and non-rural data.
    - Feature importance plot.
    """
    # Get data splits
    X_train, X_val, X_test, y_train, y_val, y_test, ct = train_validation_test_split(data_file)
    splits = rural_vs_nonrural_split(data_file)

    X_val_rural_1 = splits['X_val_rural_1']
    X_val_rural_0 = splits['X_val_rural_0']
    y_val_rural_1 = splits['y_val_rural_1']
    y_val_rural_0 = splits['y_val_rural_0']

    # Train Logistic Regression
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train, y_train)

    # Evaluate on validation set
    lr_y_valhat = lr.predict(X_val)
    lr_y_rural_valhat = lr.predict(X_val_rural_1)
    lr_y_norural_valhat = lr.predict(X_val_rural_0)

    print("Validation Results:")
    print(classification_report(y_val, lr_y_valhat, digits=4))
    acc = accuracy_score(y_val, lr_y_valhat)
    print("Validation Accuracy: %.5f" % acc)

    # Evaluate on test set
    lr_y_test = lr.predict(X_test)
    print("\nTest Results:")
    print(classification_report(y_test, lr_y_test, digits=4))
    print(f"Test Accuracy: {accuracy_score(y_test, lr_y_test):.5f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_y_test, labels=[1, 0]))

    print("\nRural schools:")
    print(classification_report(y_val_rural_1, lr_y_rural_valhat, digits=4))

    print("Non-Rural schools:")
    print(classification_report(y_val_rural_0, lr_y_norural_valhat, digits=4))

    # Feature importance
    lr_coefficients = np.abs(lr.coef_).flatten()
    feature_names = ct.get_feature_names_out()

    lr_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': lr_coefficients})
    lr_importance_df = apply_feature_name_mapping(lr_importance_df)  # Map features to human-readable names
    lr_importance_df = lr_importance_df.sort_values(by='Importance', ascending=False)

    # Filter and plot top 10 features (excluding "School State" features)
    lr_importance_df_clean = lr_importance_df[~lr_importance_df['Feature'].str.contains("School State", na=False)].head(10)

    plt.figure(figsize=(12, 15))
    plt.barh(lr_importance_df_clean['Feature'], lr_importance_df_clean['Importance'], align='center')
    plt.xlabel('Importance')
    plt.title('Feature Importance from Logistic Regression (Absolute Coefficients)')
    plt.gca().invert_yaxis()
    plt.show()

    return lr

if __name__ == "__main__":
    logistic_regression("data/data_imp_rural.csv")
