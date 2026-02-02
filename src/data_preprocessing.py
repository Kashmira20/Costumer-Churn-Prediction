import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os


class DataPreprocessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.categorical_features = None
        
    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        df = df.copy()
        
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            median_total_charges = df['TotalCharges'].median()
            df['TotalCharges'] = df['TotalCharges'].fillna(median_total_charges)
            print(f"Missing values in TotalCharges filled with median: {median_total_charges:.2f}")
        
        return df
    
    def remove_irrelevant_features(self, df):
        df = df.copy()
        
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            print("Removed customerID column")
        
        return df
    
    def encode_target(self, df):
        df = df.copy()
        
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
            print(f"Target encoded. Churn distribution:\n{df['Churn'].value_counts()}")
        
        return df
    
    def encode_categorical_features(self, df):
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        self.categorical_features = [col for col in df_encoded.columns if col not in self.numerical_features and col != 'Churn']
        
        print(f"Encoded {len(categorical_cols)} categorical features")
        print(f"Final feature count: {len(df_encoded.columns) - 1}")
        
        return df_encoded
    
    def split_features_target(self, df):
        if 'Churn' not in df.columns:
            raise ValueError("Target variable 'Churn' not found in dataframe")
        
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        
        return X, y
    
    def scale_features(self, X_train, X_test=None, fit=True):
        X_train_scaled = X_train.copy()
        
        if fit:
            X_train_scaled[self.numerical_features] = self.scaler.fit_transform(X_train[self.numerical_features])
            print("Scaler fitted on training data")
        else:
            X_train_scaled[self.numerical_features] = self.scaler.transform(X_train[self.numerical_features])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[self.numerical_features] = self.scaler.transform(X_test[self.numerical_features])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def preprocess_pipeline(self, df, is_training=True):
        df = self.handle_missing_values(df)
        df = self.remove_irrelevant_features(df)
        df = self.encode_target(df)
        df = self.encode_categorical_features(df)
        
        if is_training:
            return df
        else:
            X, y = self.split_features_target(df)
            return X, y
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {filepath}")
    
    def save_feature_names(self, feature_names, filepath='models/feature_names.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(feature_names, f)
        print(f"Feature names saved to {filepath}")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    df = preprocessor.load_data('data/raw/Telco-Customer-Churn.csv')
    
    df_processed = preprocessor.preprocess_pipeline(df, is_training=True)
    
    df_processed.to_csv('data/processed/processed_data.csv', index=False)
    print("\nPreprocessed data saved to data/processed/processed_data.csv")