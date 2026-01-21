import os
import joblib
import numpy as np
import pandas as pd
from logger import logger
from pathlib import Path
from utils import load_safe_yaml
from sklearn.linear_model import LogisticRegression

MODEL_CONFIG_PATH = Path(os.getenv("MODEL_CONFIG_PATH", 'config/model_config.yaml'))
Path("artifacts").mkdir(parents=True, exist_ok=True)

def lr_model_trainer(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Trains logistic Regression Model

    Args:
        X_train: pd.DataFrame
        y_train: pd.Series

    Returns:
        lr_model: trained model 
    """
    model_config = load_safe_yaml(MODEL_CONFIG_PATH)

    if "params" not in model_config:
        raise KeyError("Missing 'params' in model_config.yaml")

    params = model_config["params"]

    lr_model = LogisticRegression(max_iter=params["max_iter"],
                                  random_state=params.get("random_state", 42))
    
    logger.info("Logistic Regression Model training started")
    lr_model.fit(X_train, y_train)
    logger.info("Logistic Regression Model training complete")

    joblib.dump(lr_model, "artifacts/lr_model.joblib")
    logger.info("Trained Model saved as serialized file")

    return lr_model