import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'customerID': ['001', '002', '003'],
        'gender': ['Male', 'Female', 'Male'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'Yes'],
        'tenure': [12, 24, 6],
        'PhoneService': ['Yes', 'Yes', 'No'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'MonthlyCharges': [50.0, 80.0, 30.0],
        'TotalCharges': ['600', '1920', '180'],
        'Churn': ['No', 'Yes', 'No']
    })


@pytest.fixture
def preprocessor():
    return DataPreprocessor()


def test_handle_missing_values(preprocessor, sample_data):
    sample_data.loc[0, 'TotalCharges'] = ' '
    
    result = preprocessor.handle_missing_values(sample_data)
    
    assert result['TotalCharges'].isnull().sum() == 0
    assert pd.api.types.is_numeric_dtype(result['TotalCharges'])


def test_remove_irrelevant_features(preprocessor, sample_data):
    result = preprocessor.remove_irrelevant_features(sample_data)
    
    assert 'customerID' not in result.columns


def test_encode_target(preprocessor, sample_data):
    result = preprocessor.encode_target(sample_data)
    
    assert result['Churn'].dtype == np.int64 or result['Churn'].dtype == np.int32
    assert set(result['Churn'].unique()).issubset({0, 1})


def test_encode_categorical_features(preprocessor, sample_data):
    sample_data = preprocessor.encode_target(sample_data)
    result = preprocessor.encode_categorical_features(sample_data)
    
    assert result.select_dtypes(include=['object']).shape[1] == 0
    assert result.shape[1] > sample_data.shape[1]


def test_split_features_target(preprocessor, sample_data):
    sample_data = preprocessor.encode_target(sample_data)
    
    X, y = preprocessor.split_features_target(sample_data)
    
    assert 'Churn' not in X.columns
    assert y.name == 'Churn'
    assert len(X) == len(y)


def test_scale_features(preprocessor, sample_data):
    sample_data = preprocessor.handle_missing_values(sample_data)
    sample_data = preprocessor.remove_irrelevant_features(sample_data)
    sample_data = preprocessor.encode_target(sample_data)
    sample_data = preprocessor.encode_categorical_features(sample_data)
    
    X, y = preprocessor.split_features_target(sample_data)
    
    X_scaled = preprocessor.scale_features(X, fit=True)
    
    assert X_scaled[preprocessor.numerical_features].mean().abs().max() < 1e-10
    assert X_scaled.shape == X.shape


def test_full_preprocessing_pipeline(preprocessor, sample_data):
    result = preprocessor.preprocess_pipeline(sample_data, is_training=True)
    
    assert 'customerID' not in result.columns
    assert result['Churn'].dtype in [np.int64, np.int32]
    assert result.select_dtypes(include=['object']).shape[1] == 0
    assert result.isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])