import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, recall_score
import pickle
import os
from data_preprocessing import DataPreprocessor


class ChurnModelTrainer:
    
    def __init__(self):
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, filepath='data/raw/Telco-Customer-Churn.csv', test_size=0.2, random_state=42):
        df = self.preprocessor.load_data(filepath)
        
        df_processed = self.preprocessor.preprocess_pipeline(df, is_training=True)
        
        X, y = self.preprocessor.split_features_target(df_processed)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTrain set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Train churn rate: {self.y_train.mean():.2%}")
        print(f"Test churn rate: {self.y_test.mean():.2%}")
        
        self.X_train, self.X_test = self.preprocessor.scale_features(self.X_train, self.X_test, fit=True)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, class_weight='balanced', max_iter=1000, random_state=42):
        print("\nTraining Logistic Regression with balanced class weights...")
        
        self.model = LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs'
        )
        
        self.model.fit(self.X_train, self.y_train)
        
        print("Model training completed")
        
        return self.model
    
    def evaluate_model(self, threshold=0.5):
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\nModel Performance (threshold={threshold}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def cross_validate_model(self, cv=5):
        print(f"\nPerforming {cv}-Fold Cross-Validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        accuracy_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='accuracy')
        recall_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='recall')
        roc_auc_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=skf, scoring='roc_auc')
        
        print(f"Cross-Validation Results:")
        print(f"Mean Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
        print(f"Mean Recall: {recall_scores.mean():.4f} (+/- {recall_scores.std():.4f})")
        print(f"Mean ROC-AUC: {roc_auc_scores.mean():.4f} (+/- {roc_auc_scores.std():.4f})")
        
        return {
            'accuracy': accuracy_scores,
            'recall': recall_scores,
            'roc_auc': roc_auc_scores
        }
    
    def get_feature_importance(self, top_n=10):
        feature_names = self.X_train.columns
        coefficients = self.model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(feature_importance.head(top_n)[['feature', 'coefficient']])
        
        return feature_importance
    
    def save_model(self, filepath='models/logistic_model.pkl'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to {filepath}")
        
        self.preprocessor.save_scaler('models/scaler.pkl')
        self.preprocessor.save_feature_names(list(self.X_train.columns), 'models/feature_names.pkl')
    
    def load_model(self, filepath='models/logistic_model.pkl'):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        
        self.preprocessor.load_scaler('models/scaler.pkl')


def main():

    trainer = ChurnModelTrainer()
    
    trainer.load_and_prepare_data()
    
    trainer.train_model()
    
    results = trainer.evaluate_model(threshold=0.5)
    
    cv_results = trainer.cross_validate_model(cv=5)
    
    feature_importance = trainer.get_feature_importance(top_n=10)
    
    trainer.save_model()
    
if __name__ == "__main__":
    main()