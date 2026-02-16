from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import sys
import os
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_preprocessing import DataPreprocessor

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CustomerData(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=1397.47)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 1397.47
            }
        }

class PredictionResponse(BaseModel):
    churn_prediction: str
    churn_probability: float
    risk_level: str
    confidence: str
    recommendations: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str

try:
    with open('models/logistic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    preprocessor = DataPreprocessor()
    preprocessor.scaler = scaler
    
    model_loaded = True
    print("Models loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model files: {str(e)}")
    model_loaded = False

@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerData):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files are present."
        )
    
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
        
        if probability >= 0.7:
            risk_level = "High"
            confidence = "Very Confident"
        elif probability >= 0.5:
            risk_level = "Medium-High"
            confidence = "Confident"
        elif probability >= 0.3:
            risk_level = "Medium"
            confidence = "Moderate"
        else:
            risk_level = "Low"
            confidence = "High"
        
        recommendations = generate_recommendations(customer, probability)
        
        return {
            "churn_prediction": "Will Churn" if prediction == 1 else "Will Stay",
            "churn_probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "confidence": confidence,
            "recommendations": recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/batch-predict", tags=["Prediction"])
def batch_predict(customers: List[CustomerData]):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files are present."
        )
    
    if len(customers) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size limited to 100 customers"
        )
    
    results = []
    for customer in customers:
        try:
            prediction = predict_churn(customer)
            results.append({
                "customer_data": customer.dict(),
                "prediction": prediction
            })
        except Exception as e:
            results.append({
                "customer_data": customer.dict(),
                "error": str(e)
            })
    
    return {
        "total_customers": len(customers),
        "predictions": results
    }

@app.get("/model-info", tags=["Model"])
def get_model_info():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "Logistic Regression",
        "algorithm": "Balanced Logistic Regression",
        "features_count": len(feature_names),
        "performance_metrics": {
            "accuracy": 0.74,
            "recall": 0.78,
            "precision": 0.51,
            "roc_auc": 0.84
        },
        "training_info": {
            "dataset": "Telco Customer Churn",
            "samples": 7043,
            "class_weight": "balanced"
        }
    }

def generate_recommendations(customer: CustomerData, probability: float) -> List[str]:
    recommendations = []
    
    if probability >= 0.5:
        if customer.Contract == "Month-to-month":
            recommendations.append("Offer 15% discount for switching to annual contract")
        
        if customer.MonthlyCharges > 70:
            recommendations.append("Review pricing plan and offer customized package")
        
        if customer.tenure < 12:
            recommendations.append("Assign dedicated account manager for onboarding support")
        
        if customer.InternetService == "Fiber optic":
            recommendations.append("Schedule service quality check for fiber optic connection")
        
        if customer.TechSupport == "No":
            recommendations.append("Offer complimentary tech support for 3 months")
        
        if customer.OnlineSecurity == "No":
            recommendations.append("Promote security package with limited-time offer")
        
        if not recommendations:
            recommendations.append("Initiate personalized retention campaign")
            recommendations.append("Schedule customer satisfaction survey call")
    else:
        recommendations.append("Continue regular engagement and service quality")
        recommendations.append("Consider for loyalty rewards program")
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)