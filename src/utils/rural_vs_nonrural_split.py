import numpy as np
from scipy.sparse import csr_matrix
from train_validation_test_split import train_validation_test_split

def rural_vs_nonrural_split(data_file):
    """
    Further splits the train, validation, and test sets into rural and non-rural subsets.

    Parameters:
    - data_file: Path to the cleaned data CSV.

    Returns:
    - Dictionary containing rural and non-rural subsets for train, validation, and test.
    """
    X_train, X_val, X_test, y_train, y_val, y_test, ct = train_validation_test_split(data_file)

    # Extract 'is_rural' feature index
    feature_names = ct.get_feature_names_out()
    school_rural_index = list(feature_names).index('remainder__is_rural')

    # Rural and non-rural splits
    school_rural_column_train = X_train[:, school_rural_index].toarray().flatten()
    school_rural_column_val = X_val[:, school_rural_index].toarray().flatten()
    school_rural_column_test = X_test[:, school_rural_index].toarray().flatten()

    splits = {
        'X_train_rural_1': X_train[school_rural_column_train == 1],
        'X_train_rural_0': X_train[school_rural_column_train == 0],
        'y_train_rural_1': y_train[school_rural_column_train == 1],
        'y_train_rural_0': y_train[school_rural_column_train == 0],
        'X_val_rural_1': X_val[school_rural_column_val == 1],
        'X_val_rural_0': X_val[school_rural_column_val == 0],
        'y_val_rural_1': y_val[school_rural_column_val == 1],
        'y_val_rural_0': y_val[school_rural_column_val == 0],
        'X_test_rural_1': X_test[school_rural_column_test == 1],
        'X_test_rural_0': X_test[school_rural_column_test == 0],
        'y_test_rural_1': y_test[school_rural_column_test == 1],
        'y_test_rural_0': y_test[school_rural_column_test == 0]
    }

    return splits

if __name__ == "__main__":
    splits = rural_vs_nonrural_split("data/data_imp_rural.csv")
    for key, value in splits.items():
        print(f"{key} shape: {value.shape}")
