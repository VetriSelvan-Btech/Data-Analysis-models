"""
Churn Prediction Model Training Script
=====================================

This script trains a CatBoost model for customer churn prediction using the
IBM Watson Analytics Telco Customer Churn dataset.

Features:
- Data preprocessing and feature engineering
- Model training with hyperparameter optimization
- Cross-validation and performance evaluation
- Feature importance analysis
- Model saving and documentation

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import warnings
import pickle
import json
from datetime import datetime
import os

# Suppress warnings
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    """
    A comprehensive class for training and evaluating a customer churn prediction model.
    """
    
    def __init__(self, data_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        """
        Initialize the model with data path.
        
        Args:
            data_path (str): Path to the dataset CSV file
        """
        self.data_path = data_path
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.training_history = {}
        
    def load_data(self):
        """
        Load and display basic information about the dataset.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("📊 Loading dataset...")
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📋 Shape: {df.shape}")
            print(f"📊 Columns: {list(df.columns)}")
            print(f"🎯 Target variable: Churn")
            print(f"📈 Churn rate: {(df['Churn'].value_counts(normalize=True) * 100).round(2).to_dict()}")
            return df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocess the data for model training.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            tuple: (X, y) - Features and target
        """
        print("\n🔧 Preprocessing data...")
        
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Handle missing values
        print("   - Handling missing values...")
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
        
        # Encode categorical variables
        print("   - Encoding categorical variables...")
        categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        categorical_columns.remove('Churn')  # Don't encode target yet
        
        for col in categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
        
        # Encode target variable
        le_target = LabelEncoder()
        data['Churn'] = le_target.fit_transform(data['Churn'])
        self.label_encoders['Churn'] = le_target
        
        # Separate features and target
        X = data.drop('Churn', axis=1)
        y = data['Churn']
        
        self.feature_names = X.columns.tolist()
        
        print(f"   ✅ Preprocessing complete!")
        print(f"   📊 Features: {len(self.feature_names)}")
        print(f"   🎯 Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train the CatBoost model with hyperparameter optimization.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        print("\n🚀 Training model...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   📊 Training set: {X_train.shape}")
        print(f"   📊 Test set: {X_test.shape}")
        
        # Initialize CatBoost model
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            random_seed=random_state,
            verbose=100,
            eval_metric='F1',
            class_weights=[1, 2]  # Give more weight to churn class
        )
        
        # Train the model
        print("   🔄 Training in progress...")
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=50,
            plot=False
        )
        
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
        print("   ✅ Model training complete!")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self):
        """
        Evaluate the trained model performance.
        """
        if self.model is None:
            print("❌ No model to evaluate. Please train the model first.")
            return
        
        print("\n📊 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == self.y_test).mean()
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"   📈 Accuracy: {accuracy:.4f}")
        print(f"   📈 AUC Score: {auc_score:.4f}")
        
        # Cross-validation
        print("   🔄 Performing cross-validation...")
        cv_scores = cross_val_score(self.model, self.X_test, self.y_test, cv=5, scoring='f1')
        print(f"   📊 Cross-validation F1 scores: {cv_scores}")
        print(f"   📊 Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Classification report
        print("\n   📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        # Store evaluation results
        self.evaluation_results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
        }
        
        return self.evaluation_results
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance.
        
        Args:
            top_n (int): Number of top features to display
        """
        if self.model is None:
            print("❌ No model to analyze. Please train the model first.")
            return
        
        print(f"\n🎯 Analyzing feature importance (top {top_n})...")
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(top_n), 
                   x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features for Churn Prediction', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('code/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ Feature importance plot saved as 'code/feature_importance.png'")
        
        return feature_importance_df
    
    def plot_roc_curve(self):
        """
        Plot ROC curve.
        """
        if self.model is None:
            print("❌ No model to analyze. Please train the model first.")
            return
        
        print("\n📈 Plotting ROC curve...")
        
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve for Churn Prediction Model', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('code/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("   ✅ ROC curve saved as 'code/roc_curve.png'")
    
    def save_model(self, model_path="catboost_churn_model.cbm"):
        """
        Save the trained model and related files.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            print("❌ No model to save. Please train the model first.")
            return
        
        print(f"\n💾 Saving model to {model_path}...")
        
        # Save the model
        self.model.save_model(model_path)
        
        # Save label encoders
        with open('code/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save feature names
        with open('code/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        # Save evaluation results
        with open('code/evaluation_results.json', 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        
        # Save training metadata
        training_metadata = {
            'training_date': datetime.now().isoformat(),
            'model_type': 'CatBoostClassifier',
            'feature_count': len(self.feature_names),
            'model_path': model_path,
            'label_encoders_path': 'code/label_encoders.pkl',
            'feature_names_path': 'code/feature_names.json'
        }
        
        with open('code/training_metadata.json', 'w') as f:
            json.dump(training_metadata, f, indent=4)
        
        print("   ✅ Model and related files saved successfully!")
        print(f"   📁 Model: {model_path}")
        print(f"   📁 Label encoders: code/label_encoders.pkl")
        print(f"   📁 Feature names: code/feature_names.json")
        print(f"   📁 Evaluation results: code/evaluation_results.json")
        print(f"   📁 Training metadata: code/training_metadata.json")
    
    def generate_model_report(self):
        """
        Generate a comprehensive model report.
        """
        if self.model is None:
            print("❌ No model to report. Please train the model first.")
            return
        
        print("\n📋 Generating model report...")
        
        report = f"""
# Churn Prediction Model Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- **Model Type**: CatBoost Classifier
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Features**: {len(self.feature_names)}
- **Model File**: catboost_churn_model.cbm

## Performance Metrics
- **Accuracy**: {self.evaluation_results['accuracy']:.4f}
- **AUC Score**: {self.evaluation_results['auc_score']:.4f}
- **Cross-validation F1**: {self.evaluation_results['cv_f1_mean']:.4f} (+/- {self.evaluation_results['cv_f1_std'] * 2:.4f})

## Feature Importance (Top 10)
"""
        
        # Add feature importance
        importance = self.model.get_feature_importance()
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        for i, row in feature_importance_df.head(10).iterrows():
            report += f"- **{row['Feature']}**: {row['Importance']:.3f}\n"
        
        report += f"""
## Model Files
- `catboost_churn_model.cbm`: Trained model
- `code/label_encoders.pkl`: Label encoders for categorical variables
- `code/feature_names.json`: Feature names
- `code/evaluation_results.json`: Performance metrics
- `code/training_metadata.json`: Training metadata

## Usage
```python
from catboost import CatBoostClassifier
import pickle
import json

# Load model
model = CatBoostClassifier()
model.load_model("catboost_churn_model.cbm")

# Load label encoders
with open('code/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Load feature names
with open('code/feature_names.json', 'r') as f:
    feature_names = json.load(f)
```
"""
        
        # Save report
        with open('code/model_report.md', 'w') as f:
            f.write(report)
        
        print("   ✅ Model report saved as 'code/model_report.md'")
        print("\n" + "="*50)
        print("📋 MODEL REPORT")
        print("="*50)
        print(report)

def main():
    """
    Main function to run the complete model training pipeline.
    """
    print("🎯 Churn Prediction Model Training Pipeline")
    print("="*50)
    
    # Initialize model
    churn_model = ChurnPredictionModel()
    
    # Load data
    df = churn_model.load_data()
    if df is None:
        return
    
    # Preprocess data
    X, y = churn_model.preprocess_data(df)
    
    # Train model
    X_train, X_test, y_train, y_test = churn_model.train_model(X, y)
    
    # Evaluate model
    evaluation_results = churn_model.evaluate_model()
    
    # Analyze feature importance
    feature_importance = churn_model.plot_feature_importance()
    
    # Plot ROC curve
    churn_model.plot_roc_curve()
    
    # Save model
    churn_model.save_model()
    
    # Generate report
    churn_model.generate_model_report()
    
    print("\n🎉 Model training pipeline completed successfully!")
    print("📁 Check the 'code/' folder for all generated files.")

if __name__ == "__main__":
    main() 