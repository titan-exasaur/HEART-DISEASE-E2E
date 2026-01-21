from data_ingestion import *

# 1. DATA COLLECTION
download_data()
schema = load_schema(SCHEMA_PATH)
raw_df = load_data(DATA_RAW_PATH)
validate_data(raw_df, schema)

