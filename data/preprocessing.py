import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .load_data import ID_COLUMNS, TARGET_COLUMN


def remove_identifier_columns(
    X: pd.DataFrame,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove identifiers and non-predictive variables.
    """

    if id_columns is None:
        id_columns = ID_COLUMNS

    existing_columns = [col for col in id_columns if col in X.columns]

    return X.drop(columns=existing_columns)


def remove_high_missing_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Remove columns whose missing-value ratio in the training set
    is above the selected threshold.
    """

    missing_ratio = X_train.isna().mean()
    columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()

    X_train = X_train.drop(columns=columns_to_drop)
    X_val = X_val.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)

    return X_train, X_val, X_test, columns_to_drop


def create_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create train/validation/test split.

    Final proportions:
    - train: 60%
    - validation: 20%
    - test: 20%
    """

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    val_relative_size = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_preprocessing(
    X_train: pd.DataFrame,
) -> tuple[SimpleImputer, StandardScaler]:
    """
    Fit imputer and scaler using only the training set.
    """

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    scaler.fit(X_train_imputed)

    return imputer, scaler


def apply_preprocessing(
    X: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply fitted preprocessing objects to a dataset.
    """

    X_imputed = imputer.transform(X)
    X_scaled = scaler.transform(X_imputed)

    return X_scaled


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Add an intercept column for linear models.
    The intercept is not standardized.
    """

    intercept = np.ones((X.shape[0], 1))

    return np.hstack([intercept, X])


def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    missing_threshold: float = 0.30,
    random_state: int = 42,
    add_bias: bool = True,
) -> dict:
    """
    Complete preprocessing pipeline.

    Returns a dictionary with processed arrays and metadata.
    """

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(float)

    X = remove_identifier_columns(X)

    X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
        X,
        y,
        random_state=random_state,
    )

    X_train, X_val, X_test, dropped_columns = remove_high_missing_columns(
        X_train,
        X_val,
        X_test,
        threshold=missing_threshold,
    )

    feature_names = X_train.columns.tolist()

    imputer, scaler = fit_preprocessing(X_train)

    X_train_processed = apply_preprocessing(X_train, imputer, scaler)
    X_val_processed = apply_preprocessing(X_val, imputer, scaler)
    X_test_processed = apply_preprocessing(X_test, imputer, scaler)

    if add_bias:
        X_train_processed = add_intercept(X_train_processed)
        X_val_processed = add_intercept(X_val_processed)
        X_test_processed = add_intercept(X_test_processed)

        feature_names = ["intercept"] + feature_names

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
        "feature_names": feature_names,
        "dropped_columns": dropped_columns,
        "imputer": imputer,
        "scaler": scaler,
    }