"""
Model Evaluation for Churn Prediction
===================================

This script provides comprehensive evaluation of the trained churn prediction model.
It includes performance metrics, visualizations, and detailed analysis.

Features:
- Performance metrics calculation
- Confusion matrix analysis
- ROC curve and AUC analysis
- Feature importance analysis
- Model comparison and benchmarking
- Detailed evaluation reports

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, f1_score, precision_score, recall_score,
    accuracy_score, log_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from catboost import CatBoostClassifier
import pickle
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelEvaluator:
    """
    A comprehensive class for evaluating churn prediction models.
    """
    
    def __init__(self, model_path="catboost_churn_model.cbm", 
                 label_encoders_path="code/label_encoders.pkl",
                 feature_names_path="code/feature_names.json"):
        """
        Initialize the evaluator.
        
        Args:
            model_path (str): Path to the trained model
            label_encoders_path (str): Path to label encoders
            feature_names_path (str): Path to feature names
        """
        self.model_path = model_path
        self.label_encoders_path = label_encoders_path
        self.feature_names_path = feature_names_path
        self.model = None
        self.label_encoders = None
        self.feature_names = None
        self.evaluation_results = {}
        
    def load_model_and_data(self):
        """
        Load the trained model and related data.
        """
        print("📊 Loading model and data...")
        
        try:
            # Load model
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            print("   ✅ Model loaded successfully!")
            
            # Load label encoders
            with open(self.label_encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            print("   ✅ Label encoders loaded successfully!")
            
            # Load feature names
            with open(self.feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print("   ✅ Feature names loaded successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model/data: {e}")
            return False
    
    def prepare_test_data(self, test_data_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        """
        Prepare test data for evaluation.
        
        Args:
            test_data_path (str): Path to test dataset
            
        Returns:
            tuple: (X_test, y_test) - Test features and target
        """
        print("\n🔧 Preparing test data...")
        
        try:
            # Load data
            df = pd.read_csv(test_data_path)
            
            # Handle missing values
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
            
            # Encode categorical variables using saved encoders
            for col in df.columns:
                if col in self.label_encoders and col != 'Churn':
                    df[col] = self.label_encoders[col].transform(df[col])
            
            # Encode target
            if 'Churn' in self.label_encoders:
                df['Churn'] = self.label_encoders['Churn'].transform(df['Churn'])
            
            # Separate features and target
            X_test = df.drop('Churn', axis=1)
            y_test = df['Churn']
            
            # Ensure columns match feature names
            X_test = X_test[self.feature_names]
            
            print(f"   ✅ Test data prepared: {X_test.shape}")
            print(f"   🎯 Target distribution: {y_test.value_counts().to_dict()}")
            
            return X_test, y_test
            
        except Exception as e:
            print(f"❌ Error preparing test data: {e}")
            return None, None
    
    def calculate_basic_metrics(self, X_test, y_test):
        """
        Calculate basic performance metrics.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Basic metrics
        """
        print("\n📊 Calculating basic metrics...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        logloss = log_loss(y_test, y_pred_proba)
        
        # Store results
        basic_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'log_loss': logloss
        }
        
        print(f"   📈 Accuracy: {accuracy:.4f}")
        print(f"   📈 Precision: {precision:.4f}")
        print(f"   📈 Recall: {recall:.4f}")
        print(f"   📈 F1-Score: {f1:.4f}")
        print(f"   📈 AUC Score: {auc:.4f}")
        print(f"   📈 Log Loss: {logloss:.4f}")
        
        self.evaluation_results['basic_metrics'] = basic_metrics
        return basic_metrics
    
    def perform_cross_validation(self, X_test, y_test, cv=5):
        """
        Perform cross-validation evaluation.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            cv (int): Number of cross-validation folds
        """
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        
        # Define cross-validation strategy
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation for different metrics
        cv_metrics = {}
        
        for metric_name, scoring in [
            ('accuracy', 'accuracy'),
            ('precision', 'precision'),
            ('recall', 'recall'),
            ('f1', 'f1'),
            ('roc_auc', 'roc_auc')
        ]:
            scores = cross_val_score(self.model, X_test, y_test, 
                                   cv=cv_strategy, scoring=scoring)
            cv_metrics[metric_name] = {
                'scores': scores.tolist(),
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max()
            }
            
            print(f"   📊 {metric_name.capitalize()}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        self.evaluation_results['cross_validation'] = cv_metrics
        return cv_metrics
    
    def analyze_confusion_matrix(self, X_test, y_test):
        """
        Analyze confusion matrix and related metrics.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
        """
        print("\n📋 Analyzing confusion matrix...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Store results
        cm_analysis = {
            'confusion_matrix': cm.tolist(),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': specificity,
            'sensitivity': sensitivity,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate
        }
        
        print(f"   📊 True Negatives: {tn}")
        print(f"   📊 False Positives: {fp}")
        print(f"   📊 False Negatives: {fn}")
        print(f"   📊 True Positives: {tp}")
        print(f"   📊 Specificity: {specificity:.4f}")
        print(f"   📊 Sensitivity: {sensitivity:.4f}")
        
        self.evaluation_results['confusion_matrix_analysis'] = cm_analysis
        return cm_analysis
    
    def plot_confusion_matrix(self, X_test, y_test, save_path="code/confusion_matrix.png"):
        """
        Plot confusion matrix.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_path (str): Path to save the plot
        """
        print("\n📊 Plotting confusion matrix...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix - Churn Prediction Model', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        plt.ylabel('Actual', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Confusion matrix saved as '{save_path}'")
    
    def plot_roc_curve(self, X_test, y_test, save_path="code/roc_curve.png"):
        """
        Plot ROC curve.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_path (str): Path to save the plot
        """
        print("\n📈 Plotting ROC curve...")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Churn Prediction Model', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ ROC curve saved as '{save_path}'")
    
    def plot_precision_recall_curve(self, X_test, y_test, save_path="code/precision_recall_curve.png"):
        """
        Plot precision-recall curve.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            save_path (str): Path to save the plot
        """
        print("\n📊 Plotting precision-recall curve...")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curve - Churn Prediction Model', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Precision-recall curve saved as '{save_path}'")
    
    def analyze_feature_importance(self, top_n=15, save_path="code/feature_importance.png"):
        """
        Analyze and plot feature importance.
        
        Args:
            top_n (int): Number of top features to display
            save_path (str): Path to save the plot
        """
        print(f"\n🎯 Analyzing feature importance (top {top_n})...")
        
        # Get feature importance
        importance = self.model.get_feature_importance()
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Store results
        self.evaluation_results['feature_importance'] = {
            'top_features': feature_importance_df.head(top_n).to_dict('records'),
            'total_features': len(self.feature_names),
            'importance_summary': {
                'mean': feature_importance_df['Importance'].mean(),
                'std': feature_importance_df['Importance'].std(),
                'min': feature_importance_df['Importance'].min(),
                'max': feature_importance_df['Importance'].max()
            }
        }
        
        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_importance_df.head(top_n), 
                   x='Importance', y='Feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features - Churn Prediction Model', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ Feature importance plot saved as '{save_path}'")
        print(f"   📊 Top 5 features:")
        for i, row in feature_importance_df.head(5).iterrows():
            print(f"      {row['Feature']}: {row['Importance']:.3f}")
        
        return feature_importance_df
    
    def generate_evaluation_report(self, save_path="code/evaluation_report.md"):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path (str): Path to save the report
        """
        print("\n📋 Generating evaluation report...")
        
        # Create report
        report = f"""
# Churn Prediction Model Evaluation Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- **Model Type**: CatBoost Classifier
- **Model File**: {self.model_path}
- **Features**: {len(self.feature_names)}
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d')}

## Performance Metrics

### Basic Metrics
"""
        
        if 'basic_metrics' in self.evaluation_results:
            metrics = self.evaluation_results['basic_metrics']
            report += f"""
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **AUC Score**: {metrics['auc_score']:.4f}
- **Log Loss**: {metrics['log_loss']:.4f}
"""
        
        if 'cross_validation' in self.evaluation_results:
            report += "\n### Cross-Validation Results\n"
            cv_results = self.evaluation_results['cross_validation']
            for metric, results in cv_results.items():
                report += f"- **{metric.capitalize()}**: {results['mean']:.4f} (+/- {results['std'] * 2:.4f})\n"
        
        if 'confusion_matrix_analysis' in self.evaluation_results:
            report += "\n### Confusion Matrix Analysis\n"
            cm_analysis = self.evaluation_results['confusion_matrix_analysis']
            report += f"""
- **True Negatives**: {cm_analysis['true_negatives']}
- **False Positives**: {cm_analysis['false_positives']}
- **False Negatives**: {cm_analysis['false_negatives']}
- **True Positives**: {cm_analysis['true_positives']}
- **Specificity**: {cm_analysis['specificity']:.4f}
- **Sensitivity**: {cm_analysis['sensitivity']:.4f}
"""
        
        if 'feature_importance' in self.evaluation_results:
            report += "\n### Feature Importance (Top 10)\n"
            top_features = self.evaluation_results['feature_importance']['top_features'][:10]
            for feature in top_features:
                report += f"- **{feature['Feature']}**: {feature['Importance']:.3f}\n"
        
        report += f"""
## Generated Files
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `precision_recall_curve.png`: Precision-recall curve
- `feature_importance.png`: Feature importance visualization
- `evaluation_results.json`: Detailed evaluation results

## Model Assessment
"""
        
        # Add model assessment based on metrics
        if 'basic_metrics' in self.evaluation_results:
            metrics = self.evaluation_results['basic_metrics']
            
            if metrics['auc_score'] >= 0.9:
                auc_assessment = "Excellent"
            elif metrics['auc_score'] >= 0.8:
                auc_assessment = "Good"
            elif metrics['auc_score'] >= 0.7:
                auc_assessment = "Fair"
            else:
                auc_assessment = "Poor"
            
            report += f"""
- **AUC Score Assessment**: {auc_assessment} ({metrics['auc_score']:.3f})
- **Overall Performance**: The model shows {'strong' if metrics['f1_score'] > 0.7 else 'moderate'} performance in predicting customer churn
- **Recommendation**: {'Model is ready for production use' if metrics['auc_score'] > 0.8 else 'Model needs improvement before production deployment'}
"""
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report)
        
        print(f"   ✅ Evaluation report saved as '{save_path}'")
        
        return report
    
    def save_evaluation_results(self, save_path="code/evaluation_results.json"):
        """
        Save detailed evaluation results to JSON file.
        
        Args:
            save_path (str): Path to save the results
        """
        print(f"\n💾 Saving evaluation results to {save_path}...")
        
        # Add metadata
        self.evaluation_results['metadata'] = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': self.model_path,
            'feature_count': len(self.feature_names)
        }
        
        # Save to file
        with open(save_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=4)
        
        print(f"   ✅ Evaluation results saved as '{save_path}'")
    
    def run_complete_evaluation(self, test_data_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        """
        Run the complete evaluation pipeline.
        
        Args:
            test_data_path (str): Path to test dataset
        """
        print("📊 Complete Model Evaluation Pipeline")
        print("="*50)
        
        # Load model and data
        if not self.load_model_and_data():
            return
        
        # Prepare test data
        X_test, y_test = self.prepare_test_data(test_data_path)
        if X_test is None:
            return
        
        # Calculate basic metrics
        basic_metrics = self.calculate_basic_metrics(X_test, y_test)
        
        # Perform cross-validation
        cv_metrics = self.perform_cross_validation(X_test, y_test)
        
        # Analyze confusion matrix
        cm_analysis = self.analyze_confusion_matrix(X_test, y_test)
        
        # Create visualizations
        self.plot_confusion_matrix(X_test, y_test)
        self.plot_roc_curve(X_test, y_test)
        self.plot_precision_recall_curve(X_test, y_test)
        
        # Analyze feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Generate report
        report = self.generate_evaluation_report()
        
        # Save results
        self.save_evaluation_results()
        
        print("\n🎉 Evaluation pipeline completed successfully!")
        print("📁 Check the 'code/' folder for all generated files and reports.")

def main():
    """
    Main function to run the evaluation pipeline.
    """
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run complete evaluation
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main() 