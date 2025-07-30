#!/usr/bin/env python3
"""
Test script to verify the model works correctly
"""

import pandas as pd
from catboost import CatBoostClassifier

def test_model():
    """Test the model with sample data"""
    print("Testing the churn prediction model...")
    
    # Load the model
    model = CatBoostClassifier()
    model.load_model("catboost_churn_model.cbm")
    
    print(f"Model loaded successfully!")
    print(f"Expected features: {model.feature_names_}")
    
    # Create sample data with the exact features the model expects
    sample_data = pd.DataFrame([{
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 1397.475
    }])
    
    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Sample data columns: {list(sample_data.columns)}")
    
    # Make prediction
    try:
        prediction = model.predict(sample_data)[0]
        probability = model.predict_proba(sample_data)[0][1]
        
        print(f"\n✅ Prediction successful!")
        print(f"Predicted churn: {prediction}")
        print(f"Churn probability: {probability:.4f}")
        
        if prediction == 1:
            print("Result: Customer is likely to churn")
        else:
            print("Result: Customer is not likely to churn")
            
        return True
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n🎉 Model test completed successfully!")
        print("You can now run the Streamlit app with: streamlit run app.py")
    else:
        print("\n❌ Model test failed. Please check the error above.") 