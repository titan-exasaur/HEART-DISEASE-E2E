# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ------------------------------
# Define the FastAPI app
# ------------------------------
app = FastAPI(title="Heart Disease Predictor API")

# ------------------------------
# Define input schema
# ------------------------------
class HeartInput(BaseModel):
    age: float
    sex: int
    chest_pain_type: int
    bp: float
    cholesterol: float
    fbs_over_120: int
    ekg_results: float
    max_hr: float
    exercise_angina: int
    st_depression: float
    slope_of_st: int
    number_of_vessels_fluor: int
    thallium: int

# ------------------------------
# Load model artifacts ONCE
# ------------------------------
model = joblib.load("../artifacts/lr_model.joblib")
le_encoder = joblib.load("../artifacts/label_encoder.joblib")
scaler = joblib.load("../artifacts/minmax_scaler.joblib")

# ------------------------------
# Health check endpoint
# ------------------------------
@app.get("/")
def index():
    return {"message": "Heart Disease Predictor API is running"}

# ------------------------------
# Prediction endpoint
# ------------------------------
@app.post("/predict")
def predict(input_data: HeartInput):
    # Convert input to 2D array
    input_array = np.array([[ 
        input_data.age,
        input_data.sex,
        input_data.chest_pain_type,
        input_data.bp,
        input_data.cholesterol,
        input_data.fbs_over_120,
        input_data.ekg_results,
        input_data.max_hr,
        input_data.exercise_angina,
        input_data.st_depression,
        input_data.slope_of_st,
        input_data.number_of_vessels_fluor,
        input_data.thallium
    ]])

    # Scale features
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)

    # Convert back to original label
    prediction_label = le_encoder.inverse_transform(prediction)

    # Return as JSON
    return {"prediction": prediction_label[0]}
