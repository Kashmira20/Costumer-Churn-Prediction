#main.py - FastAPI version for more deployment options

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import DataPreprocessor

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

with open('models/logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

preprocessor = DataPreprocessor()
preprocessor.scaler = scaler

@app.get("/")
def read_root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make prediction",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        input_data = customer.dict()
        df = pd.DataFrame([input_data])
        
        categorical_cols = [col for col in df.columns if col in ['gender', 'Partner', 'Dependents', 
                            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']]
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        for feature in feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        df_encoded = df_encoded[feature_names]
        
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_encoded[numerical_features] = scaler.transform(df_encoded[numerical_features])
        
        probability = float(model.predict_proba(df_encoded)[0][1])
        prediction = int(probability >= 0.5)
        
        risk_level = "High Risk" if probability >= 0.6 else "Medium Risk" if probability >= 0.4 else "Low Risk"
        
        recommendations = []
        if prediction == 1:
            if customer.Contract == "Month-to-month":
                recommendations.append("Offer incentive for annual contract")
            if customer.MonthlyCharges > 70:
                recommendations.append("Review pricing and offer discount")
            if customer.tenure < 12:
                recommendations.append("Provide onboarding support")
        
        return {
            "prediction": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api.main:app --reload
# Deploy to: Railway, Render, Fly.io