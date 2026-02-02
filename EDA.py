# Customer Churn - Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load Data
df = pd.read_csv('../data/raw/Telco-Customer-Churn.csv')

print("Dataset Overview")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# Basic Info
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# Target Variable Analysis
print("\nChurn Distribution:")
print(df['Churn'].value_counts())
print(f"\nChurn Rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")

# Visualization 1: Churn Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['Churn'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'], alpha=0.7)
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Churn Status')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No', 'Yes'], rotation=0)

df['Churn'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], 
                                                autopct='%1.1f%%', 
                                                colors=['green', 'red'], 
                                                startangle=90)
axes[1].set_title('Churn Percentage', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# Numerical Features Analysis
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feature in enumerate(numerical_features):
    df[df['Churn'] == 'No'][feature].hist(ax=axes[idx], bins=30, alpha=0.6, label='No Churn', color='green')
    df[df['Churn'] == 'Yes'][feature].hist(ax=axes[idx], bins=30, alpha=0.6, label='Churn', color='red')
    axes[idx].set_title(f'{feature} Distribution by Churn', fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()

plt.tight_layout()
plt.show()

# Categorical Features Analysis
categorical_features = ['Contract', 'InternetService', 'PaymentMethod']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, feature in enumerate(categorical_features):
    churn_by_feature = pd.crosstab(df[feature], df['Churn'], normalize='index') * 100
    churn_by_feature.plot(kind='bar', ax=axes[idx], color=['green', 'red'], alpha=0.7)
    axes[idx].set_title(f'Churn Rate by {feature}', fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Percentage')
    axes[idx].legend(['No Churn', 'Churn'])
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Correlation Analysis
df_encoded = df.copy()
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

correlation_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
df_encoded[correlation_features] = df_encoded[correlation_features].apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(10, 8))
correlation_matrix = df_encoded[correlation_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Key Insights
print("\nKey Findings from EDA:")
print("="*60)
print("1. Dataset is imbalanced - more non-churners than churners")
print("2. Customers with lower tenure are more likely to churn")
print("3. Month-to-month contracts show highest churn rate")
print("4. Higher monthly charges correlate with increased churn")
print("5. Fiber optic internet users show elevated churn rates")
print("="*60)

# Save processed insights
insights = {
    'total_customers': len(df),
    'churn_count': df['Churn'].value_counts()['Yes'],
    'churn_rate': df['Churn'].value_counts(normalize=True)['Yes'],
    'avg_tenure_churners': df[df['Churn'] == 'Yes']['tenure'].mean(),
    'avg_tenure_non_churners': df[df['Churn'] == 'No']['tenure'].mean(),
}

print("\nNumerical Insights:")
for key, value in insights.items():
    print(f"{key}: {value}")