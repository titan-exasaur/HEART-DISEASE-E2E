import joblib
from src.heart_disease.logger import logger
from typing import Annotated
from fastapi import FastAPI, Request
from typing_extensions import Literal
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT/"artifacts"
BASE_DIR = Path(__file__).resolve().parent

LOGS_DIR = PROJECT_ROOT/"logs"
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR/"running_logs.log"


class InputArray(BaseModel):
    """
    Input array validator using Pydantic
    """
    age: Annotated[int, Field(gt=0, le=120)]
    sex: Annotated[Literal[0,1], Field(description="1 = Male, 0 = Female")]
    chest_pain_type: Annotated[Literal[1,2,3,4], Field(description="Chest pain type (1-4)")]
    bp: Annotated[int, Field(description="Resting blood pressure (mm Hg)", gt=0, le=200)]
    cholesterol: Annotated[int, Field(description="Serum cholesterol (mg/dl)", gt=0, le=600)]
    fbs_over_120: Annotated[Literal[0,1], Field(description="Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)")]
    ekg_results: Annotated[Literal[0,1,2], Field(description="Resting ECG results (0–2)")]
    max_hr: Annotated[int, Field(description="Maximum heart rate achieved", gt=0, le=250)]
    exercise_angina: Annotated[Literal[0,1], Field(description="Exercise induced angina (1 = Yes, 0 = No)")]
    st_depression: Annotated[float, Field(description="Enter ST Depression", ge=0.0, le=10.0)]
    slope_of_st: Annotated[Literal[1,2,3], Field(description="Enter Slope of ST (1-3)")]
    number_of_vessels: Annotated[Literal[0,1,2,3], Field(description="Number of major vessels colored by fluoroscopy (0–3)")]
    thallium: Annotated[Literal[3,6,7], Field(description="Thallium stress test result (3 = normal, 6 = fixed defect, 7 = reversible defect)")]

def load_artifacts():
    """
    Loads the model artifacts

    Args:
        None

    Returns:
        le: LabelEncoder
        scaler: MinMaxScaler
        model: trained LogisticRegression Model
    """
    scaler = joblib.load(ARTIFACTS_DIR/"scaler.joblib")
    le = joblib.load(ARTIFACTS_DIR/"label_encoder.joblib")
    model = joblib.load(ARTIFACTS_DIR/"lr_model.joblib")

    logger.info("Model Artifacts loaded successfully")

    return le, scaler, model

app = FastAPI(title="Heart Disease Detection System")
templates = Jinja2Templates(directory=BASE_DIR/"templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()

    # convert form data into pydantic model
    input_data = InputArray(
        age=int(form["age"]),
        sex=int(form["sex"]),
        chest_pain_type=int(form["chest_pain_type"]),
        bp=int(form["bp"]),
        cholesterol=int(form["cholesterol"]),
        fbs_over_120=int(form["fbs_over_120"]),
        ekg_results=int(form["ekg_results"]),
        max_hr=int(form["max_hr"]),
        exercise_angina=int(form["exercise_angina"]),
        st_depression=float(form["st_depression"]),
        slope_of_st=int(form["slope_of_st"]),
        number_of_vessels=int(form["number_of_vessels"]),
        thallium=int(form["thallium"])
    )

    # load model artifacts
    le, scaler, model = load_artifacts()

    # convert to 2-D array for prediction
    X =[[input_data.age, input_data.sex, input_data.chest_pain_type,
         input_data.bp, input_data.cholesterol, input_data.fbs_over_120,
         input_data.ekg_results, input_data.max_hr, input_data.exercise_angina,
         input_data.st_depression, input_data.slope_of_st, input_data.number_of_vessels,
         input_data.thallium]]
    
    # scale the input array
    X_Scaled = scaler.transform(X)

    # pass the scaled array through the trained model
    y_encoded = model.predict(X_Scaled)

    # invert the label encoding
    prediction = le.inverse_transform(y_encoded)[0]

    return templates.TemplateResponse("index.html",
                                      {
                                          "request": request,
                                          "prediction": prediction
                                      })