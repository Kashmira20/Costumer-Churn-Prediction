import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


def create_directory_structure():
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'reports/figures',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            open(gitkeep_path, 'a').close()
    
    print("Project directory structure created successfully")


def log_experiment(experiment_name, metrics, model_params, filepath='logs/experiments.json'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    experiment_log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiment_name': experiment_name,
        'metrics': metrics,
        'model_params': model_params
    }
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(experiment_log)
    
    with open(filepath, 'w') as f:
        json.dump(logs, f, indent=4)
    
    print(f"Experiment logged: {experiment_name}")


def calculate_business_impact(tp, tn, fp, fn, customer_value=1000, retention_cost=100):
    total_customers = tp + tn + fp + fn
    churners_identified = tp
    false_alarms = fp
    missed_churners = fn
    
    revenue_saved = churners_identified * customer_value
    retention_costs = (churners_identified + false_alarms) * retention_cost
    revenue_lost = missed_churners * customer_value
    
    net_benefit = revenue_saved - retention_costs - revenue_lost
    roi = (net_benefit / retention_costs) * 100 if retention_costs > 0 else 0
    
    impact = {
        'total_customers': total_customers,
        'churners_identified': churners_identified,
        'false_alarms': false_alarms,
        'missed_churners': missed_churners,
        'revenue_saved': revenue_saved,
        'retention_costs': retention_costs,
        'revenue_lost': revenue_lost,
        'net_benefit': net_benefit,
        'roi_percentage': roi
    }
    
    return impact


def print_business_impact(impact):
    print("\nBusiness Impact Analysis")
    print("=" * 50)
    print(f"Total Customers Analyzed: {impact['total_customers']}")
    print(f"Churners Identified: {impact['churners_identified']}")
    print(f"False Alarms: {impact['false_alarms']}")
    print(f"Missed Churners: {impact['missed_churners']}")
    print("\nFinancial Impact:")
    print(f"Revenue Saved: ${impact['revenue_saved']:,.2f}")
    print(f"Retention Costs: ${impact['retention_costs']:,.2f}")
    print(f"Revenue Lost: ${impact['revenue_lost']:,.2f}")
    print(f"Net Benefit: ${impact['net_benefit']:,.2f}")
    print(f"ROI: {impact['roi_percentage']:.2f}%")
    print("=" * 50)


def threshold_optimization(y_true, y_pred_proba, metric='recall', thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    if metric == 'recall':
        best_threshold = results_df.loc[results_df['recall'].idxmax(), 'threshold']
    elif metric == 'precision':
        best_threshold = results_df.loc[results_df['precision'].idxmax(), 'threshold']
    elif metric == 'f1':
        best_threshold = results_df.loc[results_df['f1'].idxmax(), 'threshold']
    else:
        best_threshold = results_df.loc[results_df['accuracy'].idxmax(), 'threshold']
    
    print(f"\nOptimal threshold for {metric}: {best_threshold:.2f}")
    print(results_df[results_df['threshold'] == best_threshold])
    
    return best_threshold, results_df


def generate_model_card(model_info, save_path='MODEL_CARD.md'):
    card_content = f"""# Model Card: Customer Churn Prediction

## Model Details
- **Model Type**: {model_info.get('model_type', 'Logistic Regression')}
- **Training Date**: {model_info.get('training_date', 'N/A')}
- **Framework**: Scikit-learn
- **Purpose**: Predict customer churn probability

## Intended Use
- **Primary Use**: Identify customers at risk of churning
- **Intended Users**: Marketing teams, customer success managers
- **Out-of-Scope Uses**: Not for use in critical decision-making without human review

## Training Data
- **Dataset**: Telco Customer Churn
- **Size**: {model_info.get('training_size', 'N/A')} samples
- **Features**: {model_info.get('num_features', 'N/A')} features

## Performance Metrics
- **Accuracy**: {model_info.get('accuracy', 'N/A')}
- **Recall**: {model_info.get('recall', 'N/A')}
- **Precision**: {model_info.get('precision', 'N/A')}
- **ROC-AUC**: {model_info.get('roc_auc', 'N/A')}

## Ethical Considerations
- Model predictions should be used as decision support, not sole decision maker
- Regular monitoring required to detect performance degradation
- Fairness evaluation recommended across customer segments

## Limitations
- Model trained on telecom industry data, may not generalize to other industries
- Performance may degrade with significant market changes
- Requires periodic retraining with new data
"""
    
    with open(save_path, 'w') as f:
        f.write(card_content)
    
    print(f"Model card generated: {save_path}")


if __name__ == "__main__":
    create_directory_structure()