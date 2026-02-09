import requests
import json

BASE_URL = "http://localhost:8000"

def test_root():
    print("\n1. Testing Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_health():
    print("\n2. Testing Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    print("\n3. Testing Model Info")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction_high_risk():
    print("\n4. Testing Prediction - High Risk Customer")
    customer_data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0
    }
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction_low_risk():
    print("\n5. Testing Prediction - Low Risk Customer")
    customer_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 48,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2160.0
    }
    response = requests.post(f"{BASE_URL}/predict", json=customer_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_batch_prediction():
    print("\n6. Testing Batch Prediction")
    customers = [
        {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.0,
            "TotalCharges": 255.0
        },
        {
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "Yes",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 60.0,
            "TotalCharges": 1440.0
        }
    ]
    response = requests.post(f"{BASE_URL}/batch-predict", json=customers)
    print(f"Status Code: {response.status_code}")
    print(f"Total Customers: {response.json()['total_customers']}")
    print(f"Predictions: {json.dumps(response.json()['predictions'], indent=2)}")

if __name__ == "__main__":
    print("Customer Churn API Testing")
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_prediction_high_risk()
        test_prediction_low_risk()
        test_batch_prediction()
        print("All tests completed successfully") 
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API")
        print("Make sure the API is running: uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\nError: {str(e)}")