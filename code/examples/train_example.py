"""
Example: Training the Churn Prediction Model
==========================================

This example demonstrates how to train the churn prediction model
using the provided training script.

Author: AI Assistant
Date: 2024
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import ChurnPredictionModel

def main():
    """
    Example of training the churn prediction model.
    """
    print("🎯 Churn Prediction Model Training Example")
    print("="*50)
    
    # Initialize the model
    print("📊 Initializing model...")
    model = ChurnPredictionModel()
    
    # Load data
    print("\n📁 Loading dataset...")
    df = model.load_data()
    if df is None:
        print("❌ Failed to load dataset. Please ensure the CSV file exists.")
        return
    
    # Preprocess data
    print("\n🔧 Preprocessing data...")
    X, y = model.preprocess_data(df)
    
    # Train model
    print("\n🚀 Training model...")
    X_train, X_test, y_train, y_test = model.train_model(X, y)
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    evaluation_results = model.evaluate_model()
    
    # Analyze feature importance
    print("\n🎯 Analyzing feature importance...")
    feature_importance = model.plot_feature_importance()
    
    # Plot ROC curve
    print("\n📈 Plotting ROC curve...")
    model.plot_roc_curve()
    
    # Save model
    print("\n💾 Saving model...")
    model.save_model()
    
    # Generate report
    print("\n📋 Generating model report...")
    model.generate_model_report()
    
    print("\n🎉 Training completed successfully!")
    print("📁 Check the 'code/' folder for generated files:")
    print("   - catboost_churn_model.cbm (trained model)")
    print("   - code/label_encoders.pkl (label encoders)")
    print("   - code/feature_names.json (feature names)")
    print("   - code/evaluation_results.json (evaluation results)")
    print("   - code/model_report.md (model report)")
    print("   - code/feature_importance.png (feature importance plot)")
    print("   - code/roc_curve.png (ROC curve plot)")

if __name__ == "__main__":
    main() 