import os
import yaml
import subprocess
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from logger import logger

# ------------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------------
load_dotenv()

DATA_RAW_DIR = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", "data/raw/Heart_Disease_Prediction.csv"))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", "config/schema.yaml"))

# ------------------------------------------------------------------
# Data Download
# ------------------------------------------------------------------
def download_data(output_dir: Path = DATA_RAW_DIR) -> None:
    """
    Download and unzip the heart disease dataset from Kaggle.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d", "neurocipher/heartdisease",
        "-p", str(output_dir),
        "--unzip"
    ]

    logger.info("Downloading dataset to %s", output_dir)

    try:
        subprocess.run(command, check=True)
        logger.info("Dataset downloaded successfully")
    except subprocess.CalledProcessError:
        logger.exception("Kaggle dataset download failed")
        raise

# ------------------------------------------------------------------
# Schema Loading
# ------------------------------------------------------------------
def load_schema(schema_path: Path = SCHEMA_PATH) -> dict:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path, "r") as file:
        return yaml.safe_load(file)

# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------
def load_data(data_path: Path = DATA_RAW_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.suffix == ".csv":
        return pd.read_csv(data_path)

    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)

    raise ValueError(f"Unsupported data format: {data_path.suffix}")

# ------------------------------------------------------------------
# Data Validation
# ------------------------------------------------------------------
def validate_data(raw_data: pd.DataFrame, schema: dict):
    """
    Validate dataset against schema.
    """

    df = raw_data.copy()

    columns_schema = schema.get("columns", {})

    # 1. Column existence
    missing_columns = set(columns_schema) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # 2. Row count (optional)
    expected_rows = schema.get("dataset", {}).get("row_count")
    if expected_rows and len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(df)}"
        )

    # 3. Column-level validation
    for column, rules in columns_schema.items():
        series = df[column]

        # Null check
        if not rules.get("nullable", True) and series.isnull().any():
            raise ValueError(f"Column '{column}' contains null values")

        # Type check
        expected_dtype = rules["dtype"]
        if expected_dtype == "int" and not pd.api.types.is_integer_dtype(series):
            raise TypeError(f"Column '{column}' must be int")

        if expected_dtype == "float" and not pd.api.types.is_float_dtype(series):
            raise TypeError(f"Column '{column}' must be float")

        if expected_dtype == "category":
            allowed = rules.get("allowed_values", [])
            if not series.isin(allowed).all():
                raise ValueError(f"Invalid values in '{column}'")

        # Allowed values
        if "allowed_values" in rules:
            invalid = ~series.isin(rules["allowed_values"])
            if invalid.any():
                raise ValueError(f"Invalid values found in '{column}'")

        # Range checks
        if "min" in rules and series.min() < rules["min"]:
            raise ValueError(
                f"Column '{column}' has values below {rules['min']}"
            )

        if "max" in rules and series.max() > rules["max"]:
            raise ValueError(
                f"Column '{column}' has values above {rules['max']}"
            )

    logger.info("Data validation completed successfully")
