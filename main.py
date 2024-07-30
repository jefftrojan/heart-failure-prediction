from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI(title="Heart Failure Prediction API", description="API for predicting heart failure based on clinical features")

# Load the saved model and scaler
model = load_model('models/heart_failure_model.h5')
scaler = joblib.load('models/scaler.pkl')

class PatientData(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Age of the patient")
    anaemia: int = Field(..., ge=0, le=1, description="If the patient had anaemia (0: No, 1: Yes)")
    creatinine_phosphokinase: int = Field(..., ge=0, description="Level of creatine phosphokinase in blood (mcg/L)")
    diabetes: int = Field(..., ge=0, le=1, description="If the patient was diabetic (0: No, 1: Yes)")
    ejection_fraction: int = Field(..., ge=0, le=100, description="Percentage of blood leaving the heart at each contraction")
    high_blood_pressure: int = Field(..., ge=0, le=1, description="If the patient had hypertension (0: No, 1: Yes)")
    platelets: float = Field(..., ge=0, description="Platelet count of blood (kiloplatelets/mL)")
    serum_creatinine: float = Field(..., ge=0, description="Level of serum creatinine in blood (mg/dL)")
    serum_sodium: int = Field(..., ge=0, description="Level of serum sodium in blood (mEq/L)")
    sex: int = Field(..., ge=0, le=1, description="Sex of the patient (0: Female, 1: Male)")
    smoking: int = Field(..., ge=0, le=1, description="If the patient smokes (0: No, 1: Yes)")
    time: int = Field(..., ge=0, description="Follow-up period (days)")

@app.post("/predict")
async def predict(patient: PatientData):
    # Convert input data to numpy array
    input_data = np.array([[
        patient.age, patient.anaemia, patient.creatinine_phosphokinase, patient.diabetes,
        patient.ejection_fraction, patient.high_blood_pressure, patient.platelets,
        patient.serum_creatinine, patient.serum_sodium, patient.sex, patient.smoking, patient.time
    ]])
    
    # Preprocess the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Interpret the prediction
    death_probability = float(prediction[0][0])
    death_risk = "High" if death_probability > 0.5 else "Low"
    
    return {
        "death_probability": death_probability,
        "death_risk": death_risk
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Heart Failure Prediction API"}