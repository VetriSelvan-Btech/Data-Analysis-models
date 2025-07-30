"""
Prediction Utilities for Churn Prediction Model
=============================================

This script provides utilities for making predictions with the trained churn prediction model.
It includes functions for single predictions, batch predictions, and prediction analysis.

Features:
- Single customer prediction
- Batch prediction processing
- Prediction probability analysis
- Risk level assessment
- Prediction explanation and insights

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import pickle
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    """
    A utility class for making churn predictions with the trained model.
    """
    
    def __init__(self, model_path="catboost_churn_model.cbm",
                 label_encoders_path="code/label_encoders.pkl",
                 feature_names_path="code/feature_names.json"):
        """
        Initialize the predictor.
        
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
        
    def load_model(self):
        """
        Load the trained model and related data.
        
        Returns:
            bool: True if loading successful
        """
        try:
            # Load model
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            
            # Load label encoders
            with open(self.label_encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            # Load feature names
            with open(self.feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            
            print("✅ Model and data loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def prepare_single_input(self, customer_data):
        """
        Prepare a single customer's data for prediction.
        
        Args:
            customer_data (dict): Dictionary containing customer information
            
        Returns:
            pd.DataFrame: Prepared data for prediction
        """
        # Define expected features and their default values
        expected_features = {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 70.0,
            'TotalCharges': 1500.0
        }
        
        # Update with provided data
        for key, value in customer_data.items():
            if key in expected_features:
                expected_features[key] = value
        
        # Convert to DataFrame
        input_df = pd.DataFrame([expected_features])
        
        # Handle missing values
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
        input_df['TotalCharges'].fillna(input_df['TotalCharges'].median(), inplace=True)
        
        # Encode categorical variables
        for col in input_df.columns:
            if col in self.label_encoders and col != 'Churn':
                try:
                    input_df[col] = self.label_encoders[col].transform(input_df[col])
                except:
                    # If value not in encoder, use default
                    default_value = self.label_encoders[col].transform([expected_features[col]])[0]
                    input_df[col] = default_value
        
        # Ensure correct column order
        input_df = input_df[self.feature_names]
        
        return input_df
    
    def predict_single_customer(self, customer_data):
        """
        Make prediction for a single customer.
        
        Args:
            customer_data (dict): Customer information
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        # Prepare input data
        input_data = self.prepare_single_input(customer_data)
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        probability = self.model.predict_proba(input_data)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low Risk"
            risk_color = "🟢"
        elif probability < 0.6:
            risk_level = "Medium Risk"
            risk_color = "🟡"
        else:
            risk_level = "High Risk"
            risk_color = "🔴"
        
        # Get feature importance for this prediction
        feature_importance = self.model.get_feature_importance()
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Create result
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'will_churn': bool(prediction),
            'top_features': feature_importance_df.head(5).to_dict('records'),
            'prediction_time': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, customer_data_list):
        """
        Make predictions for multiple customers.
        
        Args:
            customer_data_list (list): List of customer data dictionaries
            
        Returns:
            list: List of prediction results
        """
        if self.model is None:
            if not self.load_model():
                return None
        
        results = []
        
        for i, customer_data in enumerate(customer_data_list):
            try:
                result = self.predict_single_customer(customer_data)
                result['customer_id'] = i + 1
                results.append(result)
            except Exception as e:
                print(f"❌ Error predicting customer {i + 1}: {e}")
                results.append({
                    'customer_id': i + 1,
                    'error': str(e),
                    'prediction_time': datetime.now().isoformat()
                })
        
        return results
    
    def analyze_prediction_insights(self, customer_data, prediction_result):
        """
        Analyze prediction and provide insights.
        
        Args:
            customer_data (dict): Original customer data
            prediction_result (dict): Prediction result
            
        Returns:
            dict: Analysis insights
        """
        insights = {
            'risk_factors': [],
            'protective_factors': [],
            'recommendations': [],
            'key_insights': []
        }
        
        # Analyze based on prediction probability
        probability = prediction_result['probability']
        
        if probability > 0.7:
            insights['key_insights'].append("High churn risk detected - immediate attention required")
            insights['recommendations'].append("Implement retention strategies immediately")
            insights['recommendations'].append("Offer personalized incentives")
            insights['recommendations'].append("Assign high-priority customer service")
        elif probability > 0.5:
            insights['key_insights'].append("Moderate churn risk - proactive measures recommended")
            insights['recommendations'].append("Monitor customer behavior closely")
            insights['recommendations'].append("Consider targeted marketing campaigns")
        else:
            insights['key_insights'].append("Low churn risk - maintain current service quality")
            insights['recommendations'].append("Continue with current service standards")
            insights['recommendations'].append("Focus on customer satisfaction")
        
        # Analyze specific features
        if customer_data.get('Contract') == 'Month-to-month':
            insights['risk_factors'].append("Month-to-month contract (higher churn risk)")
        
        if customer_data.get('tenure', 0) < 12:
            insights['risk_factors'].append("Short tenure (less than 1 year)")
        elif customer_data.get('tenure', 0) > 60:
            insights['protective_factors'].append("Long tenure (over 5 years)")
        
        if customer_data.get('MonthlyCharges', 0) > 100:
            insights['risk_factors'].append("High monthly charges")
        
        if customer_data.get('InternetService') == 'Fiber optic':
            insights['risk_factors'].append("Fiber optic service (higher churn rate)")
        
        if customer_data.get('TechSupport') == 'Yes':
            insights['protective_factors'].append("Has tech support")
        
        if customer_data.get('OnlineSecurity') == 'Yes':
            insights['protective_factors'].append("Has online security")
        
        return insights
    
    def generate_prediction_report(self, customer_data, prediction_result, insights):
        """
        Generate a detailed prediction report.
        
        Args:
            customer_data (dict): Customer data
            prediction_result (dict): Prediction result
            insights (dict): Analysis insights
            
        Returns:
            str: Formatted report
        """
        report = f"""
# Customer Churn Prediction Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Prediction Summary
- **Churn Prediction**: {'Yes' if prediction_result['will_churn'] else 'No'}
- **Churn Probability**: {prediction_result['probability']:.1%}
- **Risk Level**: {prediction_result['risk_level']} {prediction_result['risk_color']}

## Customer Information
"""
        
        # Add customer information
        for key, value in customer_data.items():
            report += f"- **{key}**: {value}\n"
        
        report += f"""
## Risk Analysis

### Risk Factors
"""
        
        if insights['risk_factors']:
            for factor in insights['risk_factors']:
                report += f"- {factor}\n"
        else:
            report += "- No significant risk factors identified\n"
        
        report += f"""
### Protective Factors
"""
        
        if insights['protective_factors']:
            for factor in insights['protective_factors']:
                report += f"- {factor}\n"
        else:
            report += "- No significant protective factors identified\n"
        
        report += f"""
## Recommendations
"""
        
        for recommendation in insights['recommendations']:
            report += f"- {recommendation}\n"
        
        report += f"""
## Key Insights
"""
        
        for insight in insights['key_insights']:
            report += f"- {insight}\n"
        
        report += f"""
## Top Influential Features
"""
        
        for feature in prediction_result['top_features']:
            report += f"- **{feature['Feature']}**: {feature['Importance']:.3f}\n"
        
        return report
    
    def save_prediction_results(self, results, filename=None):
        """
        Save prediction results to file.
        
        Args:
            results (dict or list): Prediction results
            filename (str): Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"code/prediction_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"✅ Prediction results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def create_sample_customer():
    """
    Create a sample customer for testing.
    
    Returns:
        dict: Sample customer data
    """
    return {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 24,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0,
        'TotalCharges': 2000.0
    }

def main():
    """
    Main function to demonstrate prediction utilities.
    """
    print("🎯 Churn Prediction Utilities Demo")
    print("="*50)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Load model
    if not predictor.load_model():
        print("❌ Failed to load model. Please ensure model files exist.")
        return
    
    # Create sample customer
    sample_customer = create_sample_customer()
    
    print("\n📊 Sample Customer Data:")
    for key, value in sample_customer.items():
        print(f"   {key}: {value}")
    
    # Make prediction
    print("\n🔮 Making prediction...")
    prediction_result = predictor.predict_single_customer(sample_customer)
    
    if prediction_result:
        print(f"   Prediction: {'Churn' if prediction_result['will_churn'] else 'No Churn'}")
        print(f"   Probability: {prediction_result['probability']:.1%}")
        print(f"   Risk Level: {prediction_result['risk_level']} {prediction_result['risk_color']}")
        
        # Analyze insights
        insights = predictor.analyze_prediction_insights(sample_customer, prediction_result)
        
        print(f"\n📋 Key Insights:")
        for insight in insights['key_insights']:
            print(f"   - {insight}")
        
        print(f"\n💡 Recommendations:")
        for recommendation in insights['recommendations']:
            print(f"   - {recommendation}")
        
        # Generate report
        report = predictor.generate_prediction_report(sample_customer, prediction_result, insights)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"code/prediction_report_{timestamp}.md"
        with open(report_filename, 'w') as f:
            f.write(report)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        
        # Save prediction results
        predictor.save_prediction_results(prediction_result, f"code/sample_prediction_{timestamp}.json")
    
    print("\n🎉 Prediction utilities demo completed!")

if __name__ == "__main__":
    main() 