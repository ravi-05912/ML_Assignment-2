
import pandas as pd


# Target column name
TARGET_COLUMN = "diagnosis"

# Mapping of original target labels to numeric values
TARGET_MAPPING = {
    "B": 0,  # Benign
    "M": 1   # Malignant
}


def validate_dataset(df):
    """
    Check whether the dataset contains all required columns.
    """

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    # Separate feature columns from the target
    feature_columns = [
        col for col in df.columns
        if col != TARGET_COLUMN
    ]

    if len(feature_columns) != 30:
        raise ValueError(
            f"Expected 30 input features, "
            f"but found {len(feature_columns)}."
        )

    # Check for missing values
    if df.isnull().values.any():
        raise ValueError(
            "Dataset contains missing values."
        )

    return feature_columns


def prepare_data(df):
    """
    Separate input features and convert target labels.
    """

    feature_columns = validate_dataset(df)

    X = df[feature_columns].copy()

    y = df[TARGET_COLUMN].map(TARGET_MAPPING)

    if y.isnull().any():
        raise ValueError(
            "Invalid target labels. Expected B or M."
        )

    return X, y.astype(int)
