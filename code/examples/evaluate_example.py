"""
Example: Evaluating the Churn Prediction Model
===========================================

This example demonstrates how to evaluate the trained churn prediction model
using the provided evaluation script.

Author: AI Assistant
Date: 2024
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_evaluation import ModelEvaluator

def main():
    """
    Example of evaluating the churn prediction model.
    """
    print("📊 Churn Prediction Model Evaluation Example")
    print("="*50)
    
    # Initialize the evaluator
    print("📊 Initializing evaluator...")
    evaluator = ModelEvaluator()
    
    # Load model and data
    print("\n📁 Loading model and data...")
    if not evaluator.load_model_and_data():
        print("❌ Failed to load model. Please ensure model files exist.")
        print("   Required files:")
        print("   - catboost_churn_model.cbm")
        print("   - code/label_encoders.pkl")
        print("   - code/feature_names.json")
        return
    
    # Prepare test data
    print("\n🔧 Preparing test data...")
    X_test, y_test = evaluator.prepare_test_data()
    if X_test is None:
        print("❌ Failed to prepare test data.")
        return
    
    # Calculate basic metrics
    print("\n📈 Calculating basic metrics...")
    basic_metrics = evaluator.calculate_basic_metrics(X_test, y_test)
    
    # Perform cross-validation
    print("\n🔄 Performing cross-validation...")
    cv_metrics = evaluator.perform_cross_validation(X_test, y_test)
    
    # Analyze confusion matrix
    print("\n📋 Analyzing confusion matrix...")
    cm_analysis = evaluator.analyze_confusion_matrix(X_test, y_test)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    evaluator.plot_confusion_matrix(X_test, y_test)
    evaluator.plot_roc_curve(X_test, y_test)
    evaluator.plot_precision_recall_curve(X_test, y_test)
    
    # Analyze feature importance
    print("\n🎯 Analyzing feature importance...")
    feature_importance = evaluator.analyze_feature_importance()
    
    # Generate evaluation report
    print("\n📋 Generating evaluation report...")
    report = evaluator.generate_evaluation_report()
    
    # Save evaluation results
    print("\n💾 Saving evaluation results...")
    evaluator.save_evaluation_results()
    
    print("\n🎉 Evaluation completed successfully!")
    print("📁 Check the 'code/' folder for generated files:")
    print("   - code/confusion_matrix.png (confusion matrix plot)")
    print("   - code/roc_curve.png (ROC curve plot)")
    print("   - code/precision_recall_curve.png (precision-recall curve)")
    print("   - code/feature_importance.png (feature importance plot)")
    print("   - code/evaluation_results.json (evaluation results)")
    print("   - code/evaluation_report.md (evaluation report)")
    
    # Display summary
    print("\n📊 Evaluation Summary:")
    if basic_metrics:
        print(f"   Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"   Precision: {basic_metrics['precision']:.4f}")
        print(f"   Recall: {basic_metrics['recall']:.4f}")
        print(f"   F1-Score: {basic_metrics['f1_score']:.4f}")
        print(f"   AUC Score: {basic_metrics['auc_score']:.4f}")

if __name__ == "__main__":
    main() 