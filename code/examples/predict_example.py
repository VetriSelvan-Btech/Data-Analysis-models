"""
Example: Making Predictions with the Churn Prediction Model
======================================================

This example demonstrates how to make predictions using the trained churn prediction model.

Author: AI Assistant
Date: 2024
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction_utils import ChurnPredictor

def create_sample_customers():
    """
    Create sample customers for prediction examples.
    
    Returns:
        list: List of customer data dictionaries
    """
    customers = [
        {
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
        },
        {
            'gender': 'Female',
            'SeniorCitizen': 1,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 60,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Bank transfer (automatic)',
            'MonthlyCharges': 45.0,
            'TotalCharges': 2700.0
        },
        {
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 6,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
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
            'MonthlyCharges': 120.0,
            'TotalCharges': 720.0
        }
    ]
    return customers

def main():
    """
    Example of making predictions with the churn prediction model.
    """
    print("🔮 Churn Prediction Example")
    print("="*50)
    
    # Initialize predictor
    print("📊 Initializing predictor...")
    predictor = ChurnPredictor()
    
    # Load model
    print("\n📁 Loading model...")
    if not predictor.load_model():
        print("❌ Failed to load model. Please ensure model files exist.")
        print("   Required files:")
        print("   - catboost_churn_model.cbm")
        print("   - code/label_encoders.pkl")
        print("   - code/feature_names.json")
        return
    
    # Create sample customers
    print("\n👥 Creating sample customers...")
    customers = create_sample_customers()
    
    # Make predictions for each customer
    print("\n🔮 Making predictions...")
    for i, customer in enumerate(customers, 1):
        print(f"\n--- Customer {i} ---")
        
        # Display customer info
        print("📋 Customer Information:")
        for key, value in customer.items():
            print(f"   {key}: {value}")
        
        # Make prediction
        result = predictor.predict_single_customer(customer)
        
        if result:
            print(f"\n🎯 Prediction Results:")
            print(f"   Churn Prediction: {'Yes' if result['will_churn'] else 'No'}")
            print(f"   Churn Probability: {result['probability']:.1%}")
            print(f"   Risk Level: {result['risk_level']} {result['risk_color']}")
            
            # Analyze insights
            insights = predictor.analyze_prediction_insights(customer, result)
            
            print(f"\n📊 Risk Analysis:")
            if insights['risk_factors']:
                print("   Risk Factors:")
                for factor in insights['risk_factors']:
                    print(f"     - {factor}")
            
            if insights['protective_factors']:
                print("   Protective Factors:")
                for factor in insights['protective_factors']:
                    print(f"     - {factor}")
            
            print(f"\n💡 Recommendations:")
            for recommendation in insights['recommendations']:
                print(f"   - {recommendation}")
            
            print(f"\n🎯 Key Insights:")
            for insight in insights['key_insights']:
                print(f"   - {insight}")
            
            print(f"\n🔍 Top Influential Features:")
            for feature in result['top_features'][:3]:
                print(f"   - {feature['Feature']}: {feature['Importance']:.3f}")
        else:
            print("❌ Failed to make prediction")
    
    # Example of batch prediction
    print(f"\n📦 Batch Prediction Example:")
    print("Making predictions for all customers at once...")
    
    batch_results = predictor.predict_batch(customers)
    
    if batch_results:
        print(f"\n📊 Batch Results Summary:")
        churn_count = sum(1 for result in batch_results if result.get('will_churn', False))
        total_count = len(batch_results)
        
        print(f"   Total Customers: {total_count}")
        print(f"   Predicted to Churn: {churn_count}")
        print(f"   Predicted to Stay: {total_count - churn_count}")
        print(f"   Churn Rate: {churn_count/total_count:.1%}")
        
        # Save batch results
        predictor.save_prediction_results(batch_results, "code/batch_predictions_example.json")
        print("   ✅ Batch results saved to 'code/batch_predictions_example.json'")
    
    # Generate detailed report for first customer
    print(f"\n📄 Generating Detailed Report...")
    if customers and batch_results:
        first_customer = customers[0]
        first_result = batch_results[0]
        insights = predictor.analyze_prediction_insights(first_customer, first_result)
        
        report = predictor.generate_prediction_report(first_customer, first_result, insights)
        
        # Save report
        with open("code/detailed_prediction_report_example.md", 'w') as f:
            f.write(report)
        
        print("   ✅ Detailed report saved to 'code/detailed_prediction_report_example.md'")
    
    print(f"\n🎉 Prediction example completed successfully!")
    print("📁 Check the 'code/' folder for generated files:")
    print("   - code/batch_predictions_example.json (batch prediction results)")
    print("   - code/detailed_prediction_report_example.md (detailed report)")

if __name__ == "__main__":
    main() 