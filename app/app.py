import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor


class ChurnPredictor:
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.preprocessor = DataPreprocessor()
        
    def load_artifacts(self):
        try:
            with open('models/logistic_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            with open('models/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            return True
        except Exception as e:
            st.error(f"Error loading model artifacts: {str(e)}")
            return False
    
    def preprocess_input(self, input_data):
        df = pd.DataFrame([input_data])
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        categorical_cols = [col for col in df.columns if col in ['gender', 'Partner', 'Dependents', 
                            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']]
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        for feature in self.feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0
        
        df_encoded = df_encoded[self.feature_names]
        
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_encoded[numerical_features] = self.scaler.transform(df_encoded[numerical_features])
        
        return df_encoded
    
    def predict(self, input_data):
        X = self.preprocess_input(input_data)
        
        probability = self.model.predict_proba(X)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        return prediction, probability


def main():
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Customer Churn Prediction System")
    st.markdown("Predict customer churn probability and identify retention strategies")
    
    predictor = ChurnPredictor()
    
    if not predictor.load_artifacts():
        st.error("Failed to load model. Please ensure model files exist in the models/ directory")
        return
    
    st.sidebar.header("Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        st.subheader("Account Information")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", 
                                       ["Electronic check", "Mailed check", 
                                        "Bank transfer (automatic)", "Credit card (automatic)"])
    
    with col2:
        st.subheader("Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
        st.subheader("Charges")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
    
    if st.button("Predict Churn Probability", type="primary"):
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        prediction, probability = predictor.predict(input_data)
        
        st.markdown("---")
        st.header("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{probability*100:.1f}%")
        
        with col2:
            risk_level = "High Risk" if probability >= 0.6 else "Medium Risk" if probability >= 0.4 else "Low Risk"
            st.metric("Risk Level", risk_level)
        
        with col3:
            status = "Likely to Churn" if prediction == 1 else "Likely to Stay"
            st.metric("Prediction", status)
        
        if probability >= 0.5:
            st.error("Warning: This customer has high churn risk")
        else:
            st.success("This customer is likely to stay")
        
        st.markdown("---")
        st.subheader("Retention Recommendations")
        
        if probability >= 0.5:
            recommendations = []
            
            if contract == "Month-to-month":
                recommendations.append("Offer incentive to switch to annual or 2-year contract")
            
            if monthly_charges > 70:
                recommendations.append("Review pricing and consider offering discount or package deal")
            
            if tenure < 12:
                recommendations.append("Provide onboarding support and engagement programs for new customers")
            
            if internet_service == "Fiber optic":
                recommendations.append("Investigate service quality issues with fiber optic connection")
            
            if not recommendations:
                recommendations.append("Initiate personalized retention campaign")
                recommendations.append("Schedule customer satisfaction call")
            
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("Customer shows low churn risk. Continue regular engagement and service quality maintenance.")
    
    st.sidebar.markdown("---")
    st.sidebar.info("Model Information\n\nAlgorithm: Logistic Regression\nRecall: 78%\nROC-AUC: 0.84\nAccuracy: 74%")


if __name__ == "__main__":
    main()