import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_mapping import feature_name_mapping
from train_validation_test_split import train_validation_test_split

def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plots the top N feature importances for a given model.

    Parameters:
    - model: Trained model (e.g., Naive Bayes, Logistic Regression, etc.)
    - feature_names: List of feature names corresponding to model features.
    - top_n: Number of top features to display.

    Returns:
    - None
    """
    log_probabilities = model.feature_log_prob_
    importance_scores = np.exp(log_probabilities)  # Convert log-probabilities to probabilities

    # Filter features to exclude those containing "School State"
    filtered_features = [feature for feature in feature_names if "school_state" not in feature.lower()]
    filtered_indices = [i for i, feature in enumerate(feature_names) if "school_state" not in feature.lower()]

    filtered_probabilities = importance_scores[:, filtered_indices]
    filtered_features = [feature_name_mapping.get(f, f) for f in filtered_features]

    # Plot top features for each class
    for i, class_label in enumerate(model.classes_):
        sorted_indices = np.argsort(filtered_probabilities[i])[::-1]

        top_features = [filtered_features[idx] for idx in sorted_indices[:top_n]]
        top_probabilities = filtered_probabilities[i][sorted_indices[:top_n]]

        print(f"Class {class_label}:")
        for feature, score in zip(top_features, top_probabilities):
            print(f"  {feature}: {score:.4f}")

        # Create a DataFrame for plotting
        df = pd.DataFrame({'Feature': top_features, 'Importance': top_probabilities})
        df = df.sort_values(by='Importance', ascending=True)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(df['Feature'], df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title(f'Feature Importance for Class {class_label}')
        plt.show()

if __name__ == "__main__":
    # Example usage
    from naive_bayes_tuning import tune_naive_bayes
    best_nb_model, _ = tune_naive_bayes("data/data_imp_rural.csv")

    # Extract feature names from preprocessing pipeline
    _, _, _, _, _, _, feature_names = train_validation_test_split("data/data_imp_rural.csv")

    plot_feature_importance(best_nb_model, feature_names)