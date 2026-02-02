import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
import pandas as pd


class ModelEvaluator:
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
    def plot_confusion_matrix(self, threshold=0.5, save_path=None):
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_roc_curve(self, save_path=None):
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_precision_recall_curve(self, save_path=None):
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_feature_importance(self, feature_names, top_n=15, save_path=None):
        coefficients = self.model.coef_[0]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coefficients),
            'coefficient': coefficients
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in feature_importance['coefficient']]
        plt.barh(range(len(feature_importance)), feature_importance['importance'], color=colors, alpha=0.7)
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Absolute Coefficient Value')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()
        
        plt.legend(['Decreases Churn (Negative)', 'Increases Churn (Positive)'], loc='lower right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def plot_probability_distribution(self, save_path=None):
        plt.figure(figsize=(10, 6))
        
        churn_probs = self.y_pred_proba[self.y_test == 1]
        no_churn_probs = self.y_pred_proba[self.y_test == 0]
        
        plt.hist(no_churn_probs, bins=30, alpha=0.6, label='No Churn', color='green')
        plt.hist(churn_probs, bins=30, alpha=0.6, label='Churn', color='red')
        
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        plt.xlabel('Predicted Churn Probability')
        plt.ylabel('Frequency')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def generate_all_plots(self, feature_names, output_dir='reports/figures/'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating evaluation plots...")
        
        self.plot_confusion_matrix(save_path=f'{output_dir}confusion_matrix.png')
        self.plot_roc_curve(save_path=f'{output_dir}roc_curve.png')
        self.plot_precision_recall_curve(save_path=f'{output_dir}precision_recall_curve.png')
        self.plot_feature_importance(feature_names, save_path=f'{output_dir}feature_importance.png')
        self.plot_probability_distribution(save_path=f'{output_dir}probability_distribution.png')
        
        print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    from model_training import ChurnModelTrainer
    
    trainer = ChurnModelTrainer()
    trainer.load_and_prepare_data()
    trainer.train_model()
    
    evaluator = ModelEvaluator(trainer.model, trainer.X_test, trainer.y_test)
    evaluator.generate_all_plots(trainer.X_train.columns)