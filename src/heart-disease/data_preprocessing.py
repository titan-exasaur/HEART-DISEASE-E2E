import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from logger import logger
from utils import load_safe_yaml
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

Path("artifacts").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", 'config/model_config.yaml'))


def data_segregator(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split raw dataset into feature matrix and target vector.

    This function validates the expected column count, assigns
    standardized column names, and separates features from the target.

    Args:
        raw_df (pd.DataFrame): Raw input dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - X: Feature dataframe
            - y: Target series
    """
    df = raw_df.copy()
    if df.shape[1] != 14:
        raise ValueError(f"Expected 14 columns, got {df.shape[1]}")
    
    df.columns = ["age", "sex", "chest_pain_type",
                    "bp", "cholesterol", "fbs_over_120",
                    "ekg_results", "max_hr", "exercise_angina",
                    "st_depression", "slope_of_st", 
                    "number_of_vessels_fluor", "thallium",
                    "target"]
    
    X = df.drop("target", axis=1)
    y = df["target"]
    logger.info("Data segregated into features and target successfully")

    return X, y

def data_encoding(target_col: pd.Series) -> Tuple[object, np.ndarray]:
    """
    Encode the target column using label encoding.

    The fitted encoder is serialized and stored for reuse
    during inference.

    Args:
        target_col (pd.Series): Target column to encode.

    Returns:
        Tuple[LabelEncoder, np.ndarray]:
            - le: Fitted label encoder
            - y_encoded: Encoded target values
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(target_col)
    logger.info("Target Column Label Encoding successful")

    joblib.dump(le, "artifacts/label_encoder.joblib")
    logger.info("Label Encoder saved as serialized file.")

    return le, y_encoded

def data_normalization(X: pd.DataFrame) -> Tuple[object, pd.DataFrame]:
    """
    Scale numerical features using Min-Max normalization.

    The fitted scaler is saved for consistent preprocessing
    during inference.

    Args:
        X (pd.DataFrame): Input feature set.

    Returns:
        Tuple[MinMaxScaler, pd.DataFrame]:
            - scaler: Fitted MinMaxScaler
            - X_scaled_df: Scaled feature dataframe
    """
    minmax_scaler = MinMaxScaler()
    X_scaled = minmax_scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    logger.info("Data transformation complete")

    joblib.dump(minmax_scaler, "artifacts/scaler.joblib")
    logger.info("MinMax Scaler saved as serialized file")

    return minmax_scaler, X_scaled_df

def transformed_data_saver(X_scaled_df: pd.DataFrame, y_encoded: pd.Series):
    """
    Combine transformed features and encoded target and persist to disk.

    Args:
        X_scaled_df (pd.DataFrame): Scaled feature dataframe.
        y_encoded (pd.Series): Encoded target values.

    Returns:
        None
    """
    y_df = pd.DataFrame(y_encoded, columns=['transformed_y'])
    transformed_data = pd.concat([X_scaled_df, y_df], axis=1)
    transformed_data.to_csv("data/processed/transformed_data.csv", index=False)
    logger.info("Transformed data saved successfully")

def data_splitter(X_scaled_df: pd.DataFrame, y_encoded: pd.Series):
    """
    Splits transformed data into train-test partitions

    Args:
        X_scaled_df: pd.DataFrame
        y_encoded: pd.Series

    Returns:
        X_train, X_test, y_train, y_test
    """

    model_config = load_safe_yaml(MODEL_CONFIG_PATH)

    if "params" not in model_config:
        raise KeyError("Missing 'params' in model_config.yaml")

    params = model_config["params"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_encoded,
                                                    test_size=params["test_size"],
                                                    random_state=params["random_state"])

    logger.info("Data split into train-test parititons successfully")

    return X_train, X_test, y_train, y_test

def data_preprocessing(raw_df: pd.DataFrame)  -> Tuple[
    LabelEncoder,
    MinMaxScaler,
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray
]:
    """
    Execute the complete data preprocessing pipeline.

    This function performs:
    - Feature/target segregation
    - Target variable encoding
    - Feature scaling
    - Saving preprocessing artifacts and transformed dataset

    Args:
        raw_df (pd.DataFrame): Raw input dataset.

    Returns:
        Tuple[LabelEncoder, MinMaxScaler, pd.DataFrame]:
            - le: Fitted label encoder
            - scaler: Fitted MinMax scaler
            - transformed_data: Fully processed dataset
    """
    X, y = data_segregator(raw_df)

    le, y_encoded = data_encoding(y)

    scaler, X_scaled_df = data_normalization(X)

    transformed_data_saver(X_scaled_df, y_encoded)

    X_train, X_test, y_train, y_test = data_splitter(X_scaled_df, y_encoded)

    return le, scaler, X_train, X_test, y_train, y_test
