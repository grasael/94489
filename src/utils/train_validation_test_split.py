import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer

def train_validation_test_split(data_file):
    """
    Split the data into train, validation, and test sets, and preprocess features.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Returns:
    - Preprocessed train, validation, and test splits (X and y) and the column transformer.
    """
    # Load data
    data = pd.read_csv(data_file).drop(columns=["need_statement"])

    y = data['not_fully_funded']
    X = data.drop('not_fully_funded', axis=1)

    # Train-Validation-Test Split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=2, stratify=y_train_val
    )

    # Preprocessing steps
    ohe = OneHotEncoder(handle_unknown='ignore')

    vectorizer_temp = CountVectorizer(stop_words='english')
    built_in_stop_words = vectorizer_temp.get_stop_words()
    custom_stop_words = ['my', 'students', 'need']
    combined_stop_words = list(set(built_in_stop_words).union(custom_stop_words))

    vectorizer_title = CountVectorizer(stop_words='english')
    ct = make_column_transformer(
        (ohe, ['resource_type', 'school_state', 'grade_level']),
        (vectorizer_title, 'title'),
        remainder='passthrough'
    )

    # Clean text data
    for df in [X_train, X_val, X_test]:
        df['title'] = df['title'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

    # Fit and transform data
    ct.fit(X_train)
    X_train = ct.transform(X_train)
    X_val = ct.transform(X_val)
    X_test = ct.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, ct

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, ct = train_validation_test_split("data/data_imp_rural.csv")
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
