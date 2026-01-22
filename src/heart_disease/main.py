from data_ingestion import *
from data_preprocessing import data_preprocessing
from model_trainer import lr_model_trainer
from model_evaluator import model_evaluator

# 1. DATA COLLECTION
download_data()
schema = load_schema(SCHEMA_PATH)
raw_df = load_data(DATA_RAW_PATH)
validate_data(raw_df, schema)

# 2. DATA PREPROCESSING
le, scaler, X_train, X_test, y_train, y_test = data_preprocessing(raw_df)

# 3. MODEL TRAINING
trained_model = lr_model_trainer(X_train, y_train)

# 4. MODEL EVALUATION
model_evaluator(X_test, y_test, le, trained_model)