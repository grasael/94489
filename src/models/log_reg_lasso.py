import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from train_validation_test_split import train_validation_test_split  # Preprocessing split
from rural_vs_nonrural_split import rural_vs_nonrural_split  # Rural/Non-Rural split
from feature_mapping import apply_feature_name_mapping  # Feature name mapping
from src.utils.train_validation_test_split import train_validation_test_split
from src.evaluation.feature_importance import plot_feature_importance

def logistic_regression_lasso(data_file):
    """
    Train and evaluate Logistic Regression with LASSO regularization.

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

    # Extract and vectorize titles
    vectorizer = CountVectorizer(stop_words='english')
    titles = pd.concat([pd.DataFrame(X_train.todense())[0], pd.DataFrame(X_val.todense())[0], pd.DataFrame(X_test.todense())[0]])
    X_titles_train = vectorizer.fit_transform(X_train[:, 0].toarray())
    X_titles_val = vectorizer.transform(X_val[:, 0].toarray())
    X_titles_test = vectorizer.transform(X_test[:, 0].toarray())

    print(f"Total number of features in title vectors: {X_titles_train.shape[1]}")

    # Combine title vectors with other features (if applicable)
    X_train_combined = np.hstack((X_train, X_titles_train.toarray()))
    X_val_combined = np.hstack((X_val, X_titles_val.toarray()))
    X_test_combined = np.hstack((X_test, X_titles_test.toarray()))

    # Scale data
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_val_scaled = scaler.transform(X_val_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    # Scale rural/non-rural subsets
    X_val_rural_1_scaled = scaler.transform(np.hstack((X_val_rural_1, X_titles_val.toarray())))
    X_val_rural_0_scaled = scaler.transform(np.hstack((X_val_rural_0, X_titles_val.toarray())))

    # Train Logistic Regression with LASSO
    logistic_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, max_iter=1000)
    logistic_lasso.fit(X_train_scaled, y_train)

    # Evaluate on validation set
    y_val_lasso = logistic_lasso.predict(X_val_scaled)
    print("Validation Results:")
    print(classification_report(y_val, y_val_lasso, digits=4))

    # Evaluate on rural validation set
    y_rural_val_lasso = logistic_lasso.predict(X_val_rural_1_scaled)
    print("\nRural Validation Results:")
    print(classification_report(y_val_rural_1, y_rural_val_lasso, digits=4))

    # Evaluate on non-rural validation set
    y_norural_val_lasso = logistic_lasso.predict(X_val_rural_0_scaled)
    print("\nNon-Rural Validation Results:")
    print(classification_report(y_val_rural_0, y_norural_val_lasso, digits=4))

    # Feature importance
    lasso_coefficients = np.abs(logistic_lasso.coef_).flatten()
    feature_names_ct = ct.get_feature_names_out()
    feature_names_titles = vectorizer.get_feature_names_out()
    combined_feature_names = list(feature_names_ct) + list(feature_names_titles)

    importance_df = pd.DataFrame({'Feature': combined_feature_names, 'Importance': lasso_coefficients})
    importance_df = apply_feature_name_mapping(importance_df)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Display top 10 features
    print("\nTop 10 Features:")
    print(importance_df.head(10))

    return logistic_lasso

if __name__ == "__main__":
    logistic_regression_lasso("data/data_imp_rural.csv")
