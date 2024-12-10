import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
    - file_path: str, path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def plot_correlation_heatmap(data, readable_names):
    """
    Plot a correlation heatmap for the numerical columns.

    Parameters:
    - data: pd.DataFrame, input dataset.
    - readable_names: dict, mapping of column names to readable names.

    Returns:
    - None
    """
    data_readable = data.rename(columns=readable_names)
    plt.figure(figsize=(10, 6))
    correlation_matrix = data_readable[list(readable_names.values())].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.title('Correlation Heatmap')
    plt.show()

def plot_numerical_distributions(data, numerical_cols):
    """
    Plot distributions for numerical columns.

    Parameters:
    - data: pd.DataFrame, input dataset.
    - numerical_cols: list, list of numerical column names.

    Returns:
    - None
    """
    for col in numerical_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(data[col], kde=False, bins=10)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()

def plot_categorical_counts(data, categorical_cols):
    """
    Plot count plots for categorical columns.

    Parameters:
    - data: pd.DataFrame, input dataset.
    - categorical_cols: list, list of categorical column names.

    Returns:
    - None
    """
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(x=data[col], order=data[col].value_counts().index)
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

def plot_specific_distributions(data):
    """
    Plot specific distributions for predefined columns.

    Parameters:
    - data: pd.DataFrame, input dataset.

    Returns:
    - None
    """
    columns = {
        'title_word_count': 'Title Word Count',
        'description_word_count': 'Description Word Count',
        'essay_word_count': 'Essay Word Count',
    }
    for col, title in columns.items():
        plt.figure(figsize=(8, 5))
        sns.histplot(data[col], bins=10)
        plt.title(f'Distribution of {title}')
        plt.xlabel(title)
        plt.ylabel('Frequency')
        plt.show()

if __name__ == "__main__":
    file_path = "data/data_imp.csv"
    data = load_data(file_path)

    print(data.head())
    print(data.not_fully_funded.value_counts())
    print(data.describe())

    numerical_cols = ['students_reached', 'fulfillment_labor_materials', 'title_word_count',
                      'description_word_count', 'need_word_count', 'essay_word_count']
    readable_names = {
        'students_reached': 'Students Reached',
        'fulfillment_labor_materials': 'Fulfillment Cost',
        'title_word_count': 'Title Word Count',
        'description_word_count': 'Description Word Count',
        'need_word_count': 'Need Word Count',
        'essay_word_count': 'Essay Word Count'
    }
    categorical_cols = ['resource_type', 'school_state', 'grade_level']

    # Perform EDA
    plot_correlation_heatmap(data, readable_names)
    plot_numerical_distributions(data, numerical_cols)
    plot_categorical_counts(data, categorical_cols)
    plot_specific_distributions(data)
