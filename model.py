# -*- coding: utf-8 -*-
"""
Customer Churn Prediction Model
Improved version with better precision and accuracy
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """Load and preprocess the dataset"""
    print("Loading and preprocessing data...")
    
    df = pd.read_csv(file_path)
    
    # Remove customer ID
    df.drop('customerID', axis=1, inplace=True)
    
    # Handle TotalCharges - convert to numeric and fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Convert target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Feature engineering for better model performance
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['New', 'Short', 'Medium', 'Long'])
    df['MonthlyChargesGroup'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 105, 200], labels=['Low', 'Medium', 'High', 'Very High'])
    df['TotalChargesGroup'] = pd.cut(df['TotalCharges'], bins=[0, 1000, 2000, 4000, 10000], labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Create interaction features
    df['Contract_Monthly'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['Internet_Fiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['Payment_Electronic'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    
    # Create composite features
    df['ServiceCount'] = (
        (df['PhoneService'] == 'Yes').astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['OnlineBackup'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int) +
        (df['TechSupport'] == 'Yes').astype(int) +
        (df['StreamingTV'] == 'Yes').astype(int) +
        (df['StreamingMovies'] == 'Yes').astype(int)
    )
    
    return df

def train_improved_model(X, y, cat_features):
    """Train an improved CatBoost model with better hyperparameters"""
    print("Training improved model...")
    
    # Enhanced hyperparameters for better precision and accuracy
    model = CatBoostClassifier(
        iterations=1000,           # More iterations for better convergence
        learning_rate=0.03,        # Lower learning rate for better generalization
        depth=8,                   # Slightly deeper trees
        loss_function='Logloss',
        eval_metric='F1',          # Focus on F1 score for balanced precision/recall
        cat_features=cat_features,
        class_weights=[1, 3],      # Higher weight for churn class to improve recall
        l2_leaf_reg=3,             # L2 regularization to prevent overfitting
        random_strength=0.8,       # Randomization for better generalization
        bagging_temperature=0.8,   # Bagging for ensemble effect
        border_count=254,          # More splits for better precision
        verbose=100,
        early_stopping_rounds=50,  # Early stopping to prevent overfitting
        random_state=42
    )
    
    return model

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Comprehensive model evaluation"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"CV F1 mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'], 
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    print("\nPlotting feature importance...")
    
    importances = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(data=feature_importance_df.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title("Top 15 Feature Importances - Improved CatBoost Model")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance_df

def main():
    """Main function to run the complete pipeline"""
    print("Starting Customer Churn Prediction Model Training")
    print("="*60)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Prepare features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical features
    cat_features = X.select_dtypes(include='object').columns.tolist()
    print(f"Categorical features: {cat_features}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Churn rate in training set: {y_train.mean():.3f}")
    print(f"Churn rate in test set: {y_test.mean():.3f}")
    
    # Train model
    model = train_improved_model(X_train, y_train, cat_features)
    
    # Fit the model
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
        plot=True
    )
    
    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Plot feature importance
    feature_importance_df = plot_feature_importance(model, X.columns)
    
    # Save model
    model.save_model("catboost_churn_model.cbm")
    print(f"\nModel saved as 'catboost_churn_model.cbm'")
    
    # Test prediction on a sample
    sample = X_test.iloc[[0]]
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]
    print(f"\nSample prediction test:")
    print(f"Predicted churn: {prediction}")
    print(f"Churn probability: {probability:.4f}")
    
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()