"""
Data Preprocessing for Churn Prediction Model
===========================================

This script handles all data preprocessing tasks for the customer churn prediction model.
It includes data cleaning, feature engineering, and preparation for model training.

Features:
- Data loading and exploration
- Missing value handling
- Categorical variable encoding
- Feature engineering
- Data validation and quality checks

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive class for preprocessing customer churn data.
    """
    
    def __init__(self, data_path="WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        """
        Initialize the preprocessor.
        
        Args:
            data_path (str): Path to the raw dataset
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.label_encoders = {}
        self.feature_info = {}
        self.preprocessing_report = {}
        
    def load_data(self):
        """
        Load the raw dataset and perform initial exploration.
        
        Returns:
            pd.DataFrame: Raw dataset
        """
        print("📊 Loading dataset...")
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"📋 Shape: {self.raw_data.shape}")
            print(f"📊 Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return self.raw_data
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
    
    def explore_data(self):
        """
        Perform comprehensive data exploration.
        """
        if self.raw_data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n🔍 Data Exploration")
        print("="*50)
        
        # Basic info
        print("📋 Dataset Information:")
        print(f"   - Shape: {self.raw_data.shape}")
        print(f"   - Columns: {list(self.raw_data.columns)}")
        print(f"   - Data types: {self.raw_data.dtypes.value_counts().to_dict()}")
        
        # Missing values
        print("\n📊 Missing Values:")
        missing_data = self.raw_data.isnull().sum()
        missing_percent = (missing_data / len(self.raw_data)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percent': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Target variable analysis
        print("\n🎯 Target Variable Analysis:")
        churn_counts = self.raw_data['Churn'].value_counts()
        churn_percent = self.raw_data['Churn'].value_counts(normalize=True) * 100
        print(f"   - Churn distribution: {churn_counts.to_dict()}")
        print(f"   - Churn percentage: {churn_percent.to_dict()}")
        
        # Numerical variables
        print("\n📈 Numerical Variables Summary:")
        numerical_cols = self.raw_data.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            print(self.raw_data[numerical_cols].describe())
        
        # Categorical variables
        print("\n📝 Categorical Variables Summary:")
        categorical_cols = self.raw_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            unique_count = self.raw_data[col].nunique()
            print(f"   - {col}: {unique_count} unique values")
        
        # Store exploration results
        self.preprocessing_report['exploration'] = {
            'shape': self.raw_data.shape,
            'columns': list(self.raw_data.columns),
            'data_types': self.raw_data.dtypes.value_counts().to_dict(),
            'missing_values': missing_df.to_dict(),
            'churn_distribution': churn_counts.to_dict(),
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols
        }
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset.
        """
        if self.raw_data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n🔧 Handling missing values...")
        
        # Create a copy for processing
        data = self.raw_data.copy()
        
        # Check for missing values
        missing_data = data.isnull().sum()
        if missing_data.sum() == 0:
            print("   ✅ No missing values found!")
            return data
        
        print(f"   📊 Found missing values in {missing_data[missing_data > 0].count()} columns")
        
        # Handle specific columns
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                missing_count = data[col].isnull().sum()
                missing_percent = (missing_count / len(data)) * 100
                print(f"   - {col}: {missing_count} missing values ({missing_percent:.2f}%)")
                
                # Handle based on data type
                if data[col].dtype in ['int64', 'float64']:
                    # Numerical column - use median
                    median_val = data[col].median()
                    data[col].fillna(median_val, inplace=True)
                    print(f"     → Filled with median: {median_val}")
                else:
                    # Categorical column - use mode
                    mode_val = data[col].mode()[0]
                    data[col].fillna(mode_val, inplace=True)
                    print(f"     → Filled with mode: {mode_val}")
        
        # Verify no missing values remain
        remaining_missing = data.isnull().sum().sum()
        if remaining_missing == 0:
            print("   ✅ All missing values handled successfully!")
        else:
            print(f"   ⚠️ Warning: {remaining_missing} missing values still remain")
        
        return data
    
    def handle_data_types(self, data):
        """
        Convert data types to appropriate formats.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with corrected data types
        """
        print("\n🔧 Handling data types...")
        
        # Convert TotalCharges to numeric
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            # Fill any resulting NaN values with median
            median_charges = data['TotalCharges'].median()
            data['TotalCharges'].fillna(median_charges, inplace=True)
            print(f"   ✅ Converted TotalCharges to numeric (median: {median_charges:.2f})")
        
        # Convert SeniorCitizen to categorical if it's numeric
        if 'SeniorCitizen' in data.columns and data['SeniorCitizen'].dtype in ['int64', 'float64']:
            data['SeniorCitizen'] = data['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
            print("   ✅ Converted SeniorCitizen to categorical")
        
        print("   ✅ Data type handling complete!")
        return data
    
    def encode_categorical_variables(self, data):
        """
        Encode categorical variables using label encoding.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        print("\n🔧 Encoding categorical variables...")
        
        # Get categorical columns (excluding target)
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')  # Don't encode target yet
        
        print(f"   📊 Found {len(categorical_cols)} categorical variables to encode")
        
        # Encode each categorical variable
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le
            
            # Store encoding mapping
            unique_values = le.classes_
            encoded_values = le.transform(unique_values)
            mapping = dict(zip(encoded_values, unique_values))
            
            self.feature_info[col] = {
                'type': 'categorical',
                'unique_count': len(unique_values),
                'encoding_mapping': mapping
            }
            
            print(f"   ✅ Encoded {col}: {len(unique_values)} unique values")
        
        # Encode target variable separately
        if 'Churn' in data.columns:
            le_target = LabelEncoder()
            data['Churn'] = le_target.fit_transform(data['Churn'])
            self.label_encoders['Churn'] = le_target
            
            target_mapping = dict(zip(le_target.transform(le_target.classes_), le_target.classes_))
            self.feature_info['Churn'] = {
                'type': 'target',
                'unique_count': len(le_target.classes_),
                'encoding_mapping': target_mapping
            }
            
            print(f"   ✅ Encoded target variable 'Churn': {target_mapping}")
        
        print("   ✅ Categorical encoding complete!")
        return data
    
    def create_features(self, data):
        """
        Create additional features for better model performance.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with additional features
        """
        print("\n🔧 Creating additional features...")
        
        # Create tenure groups
        data['tenure_group'] = pd.cut(data['tenure'], 
                                     bins=[0, 12, 24, 36, 48, 60, 72], 
                                     labels=['0-12', '13-24', '25-36', '37-48', '49-60', '60+'])
        
        # Create monthly charges groups
        data['monthly_charges_group'] = pd.cut(data['MonthlyCharges'], 
                                              bins=[0, 30, 60, 90, 120, 150, 200], 
                                              labels=['0-30', '31-60', '61-90', '91-120', '121-150', '150+'])
        
        # Create total charges groups
        data['total_charges_group'] = pd.cut(data['TotalCharges'], 
                                            bins=[0, 1000, 2000, 3000, 4000, 5000, 10000], 
                                            labels=['0-1K', '1K-2K', '2K-3K', '3K-4K', '4K-5K', '5K+'])
        
        # Calculate average monthly charges
        data['avg_monthly_charges'] = data['TotalCharges'] / data['tenure']
        data['avg_monthly_charges'].fillna(data['MonthlyCharges'], inplace=True)
        
        # Create service count features
        service_columns = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                          'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Count active services
        data['total_services'] = 0
        for col in service_columns:
            if col in data.columns:
                # Count services that are not 'No' or 'No internet service'
                data['total_services'] += ((data[col] != 0) & (data[col] != 2)).astype(int)
        
        # Create contract duration feature
        contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        data['contract_duration'] = data['Contract'].map(contract_mapping)
        
        # Encode the new categorical features
        new_categorical_cols = ['tenure_group', 'monthly_charges_group', 'total_charges_group']
        for col in new_categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
                
                unique_values = le.classes_
                encoded_values = le.transform(unique_values)
                mapping = dict(zip(encoded_values, unique_values))
                
                self.feature_info[col] = {
                    'type': 'engineered_categorical',
                    'unique_count': len(unique_values),
                    'encoding_mapping': mapping
                }
        
        print("   ✅ Feature engineering complete!")
        print(f"   📊 Added {len(new_categorical_cols) + 4} new features")
        
        return data
    
    def validate_data(self, data):
        """
        Validate the processed data for quality and consistency.
        
        Args:
            data (pd.DataFrame): Processed data
            
        Returns:
            bool: True if validation passes
        """
        print("\n🔍 Validating processed data...")
        
        validation_results = {}
        
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        validation_results['no_missing_values'] = missing_count == 0
        print(f"   ✅ No missing values: {validation_results['no_missing_values']}")
        
        # Check for infinite values
        infinite_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        validation_results['no_infinite_values'] = infinite_count == 0
        print(f"   ✅ No infinite values: {validation_results['no_infinite_values']}")
        
        # Check data types
        validation_results['appropriate_data_types'] = True
        for col in data.columns:
            if col == 'Churn':
                continue
            if data[col].dtype == 'object':
                validation_results['appropriate_data_types'] = False
                print(f"   ⚠️ Warning: {col} is still object type")
        
        # Check target variable
        if 'Churn' in data.columns:
            unique_targets = data['Churn'].unique()
            validation_results['valid_target'] = len(unique_targets) == 2
            print(f"   ✅ Valid target variable: {validation_results['valid_target']}")
        
        # Overall validation
        all_valid = all(validation_results.values())
        print(f"   📊 Overall validation: {'✅ PASSED' if all_valid else '❌ FAILED'}")
        
        self.preprocessing_report['validation'] = validation_results
        return all_valid
    
    def prepare_for_training(self, data):
        """
        Prepare the final dataset for model training.
        
        Args:
            data (pd.DataFrame): Processed data
            
        Returns:
            tuple: (X, y) - Features and target
        """
        print("\n🚀 Preparing data for training...")
        
        # Separate features and target
        X = data.drop('Churn', axis=1)
        y = data['Churn']
        
        # Store feature information
        self.preprocessing_report['final_features'] = {
            'feature_count': len(X.columns),
            'feature_names': list(X.columns),
            'target_distribution': y.value_counts().to_dict()
        }
        
        print(f"   📊 Final dataset shape: {X.shape}")
        print(f"   🎯 Target distribution: {y.value_counts().to_dict()}")
        print(f"   📋 Features: {len(X.columns)}")
        
        return X, y
    
    def save_preprocessing_info(self, output_dir="code"):
        """
        Save preprocessing information and metadata.
        
        Args:
            output_dir (str): Directory to save files
        """
        print(f"\n💾 Saving preprocessing information to {output_dir}...")
        
        # Save label encoders
        import pickle
        with open(f'{output_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Save feature information
        with open(f'{output_dir}/feature_info.json', 'w') as f:
            json.dump(self.feature_info, f, indent=4)
        
        # Save preprocessing report
        self.preprocessing_report['preprocessing_date'] = datetime.now().isoformat()
        with open(f'{output_dir}/preprocessing_report.json', 'w') as f:
            json.dump(self.preprocessing_report, f, indent=4)
        
        print("   ✅ Preprocessing information saved successfully!")
        print(f"   📁 Label encoders: {output_dir}/label_encoders.pkl")
        print(f"   📁 Feature info: {output_dir}/feature_info.json")
        print(f"   📁 Preprocessing report: {output_dir}/preprocessing_report.json")
    
    def run_complete_preprocessing(self):
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            tuple: (X, y) - Processed features and target
        """
        print("🔧 Complete Data Preprocessing Pipeline")
        print("="*50)
        
        # Load data
        raw_data = self.load_data()
        if raw_data is None:
            return None, None
        
        # Explore data
        self.explore_data()
        
        # Handle missing values
        data = self.handle_missing_values()
        
        # Handle data types
        data = self.handle_data_types(data)
        
        # Create additional features
        data = self.create_features(data)
        
        # Encode categorical variables
        data = self.encode_categorical_variables(data)
        
        # Validate data
        if not self.validate_data(data):
            print("❌ Data validation failed!")
            return None, None
        
        # Prepare for training
        X, y = self.prepare_for_training(data)
        
        # Save preprocessing information
        self.save_preprocessing_info()
        
        # Store processed data
        self.processed_data = data
        
        print("\n🎉 Preprocessing pipeline completed successfully!")
        return X, y

def main():
    """
    Main function to run the preprocessing pipeline.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Run complete preprocessing
    X, y = preprocessor.run_complete_preprocessing()
    
    if X is not None and y is not None:
        print(f"\n📊 Final processed data:")
        print(f"   - Features shape: {X.shape}")
        print(f"   - Target shape: {y.shape}")
        print(f"   - Feature names: {list(X.columns)}")
        print(f"   - Target distribution: {y.value_counts().to_dict()}")
    else:
        print("❌ Preprocessing failed!")

if __name__ == "__main__":
    main() 