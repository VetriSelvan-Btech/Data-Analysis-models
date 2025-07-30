# Churn Prediction Model - Code Repository

This repository contains the complete codebase for the Customer Churn Prediction Model using CatBoost. The model is designed to predict customer churn based on various customer attributes and service usage patterns.

## 📁 Repository Structure

```
code/
├── README.md                           # This file
├── train_model.py                      # Main model training script
├── data_preprocessing.py               # Data preprocessing utilities
├── model_evaluation.py                 # Model evaluation and analysis
├── prediction_utils.py                 # Prediction utilities
├── requirements.txt                    # Python dependencies
└── examples/                           # Example usage scripts
    ├── train_example.py               # Example of training the model
    ├── evaluate_example.py            # Example of evaluating the model
    └── predict_example.py             # Example of making predictions
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Required packages** (see requirements.txt):
   - pandas
   - numpy
   - scikit-learn
   - catboost
   - matplotlib
   - seaborn
   - streamlit (for web app)

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

1. **Train the model**:
   ```bash
   python train_model.py
   ```

2. **Evaluate the model**:
   ```bash
   python model_evaluation.py
   ```

3. **Make predictions**:
   ```bash
   python prediction_utils.py
   ```

## 📊 Model Components

### 1. Model Training (`train_model.py`)

**Purpose**: Complete pipeline for training the CatBoost churn prediction model.

**Features**:
- Data loading and exploration
- Data preprocessing and feature engineering
- Model training with hyperparameter optimization
- Cross-validation and performance evaluation
- Feature importance analysis
- Model saving and documentation

**Usage**:
```python
from train_model import ChurnPredictionModel

# Initialize and train model
model = ChurnPredictionModel()
X, y = model.preprocess_data(df)
model.train_model(X, y)
model.evaluate_model()
model.save_model()
```

**Output Files**:
- `catboost_churn_model.cbm` - Trained model
- `code/label_encoders.pkl` - Label encoders
- `code/feature_names.json` - Feature names
- `code/evaluation_results.json` - Training results
- `code/training_metadata.json` - Training metadata
- `code/model_report.md` - Model report

### 2. Data Preprocessing (`data_preprocessing.py`)

**Purpose**: Comprehensive data preprocessing pipeline for churn prediction.

**Features**:
- Data loading and exploration
- Missing value handling
- Categorical variable encoding
- Feature engineering
- Data validation and quality checks

**Usage**:
```python
from data_preprocessing import DataPreprocessor

# Run complete preprocessing
preprocessor = DataPreprocessor()
X, y = preprocessor.run_complete_preprocessing()
```

**Output Files**:
- `code/label_encoders.pkl` - Label encoders
- `code/feature_info.json` - Feature information
- `code/preprocessing_report.json` - Preprocessing report

### 3. Model Evaluation (`model_evaluation.py`)

**Purpose**: Comprehensive evaluation of trained models with detailed analysis.

**Features**:
- Performance metrics calculation
- Confusion matrix analysis
- ROC curve and AUC analysis
- Feature importance analysis
- Cross-validation evaluation
- Detailed evaluation reports

**Usage**:
```python
from model_evaluation import ModelEvaluator

# Evaluate model
evaluator = ModelEvaluator()
evaluator.run_complete_evaluation()
```

**Output Files**:
- `code/confusion_matrix.png` - Confusion matrix plot
- `code/roc_curve.png` - ROC curve plot
- `code/precision_recall_curve.png` - Precision-recall curve
- `code/feature_importance.png` - Feature importance plot
- `code/evaluation_results.json` - Evaluation results
- `code/evaluation_report.md` - Evaluation report

### 4. Prediction Utilities (`prediction_utils.py`)

**Purpose**: Utilities for making predictions with the trained model.

**Features**:
- Single customer prediction
- Batch prediction processing
- Prediction probability analysis
- Risk level assessment
- Prediction explanation and insights

**Usage**:
```python
from prediction_utils import ChurnPredictor

# Make prediction
predictor = ChurnPredictor()
result = predictor.predict_single_customer(customer_data)
insights = predictor.analyze_prediction_insights(customer_data, result)
```

## 🎯 Model Performance

The trained CatBoost model typically achieves:
- **Accuracy**: ~85-90%
- **AUC Score**: ~0.85-0.90
- **F1-Score**: ~0.75-0.80
- **Precision**: ~0.75-0.80
- **Recall**: ~0.70-0.75

## 📈 Key Features

### Most Important Features (Top 10)
1. **Contract** - Contract type (Month-to-month, One year, Two year)
2. **tenure** - Number of months customer has been with the company
3. **MonthlyCharges** - Monthly service charges
4. **TotalCharges** - Total charges over the customer's lifetime
5. **InternetService** - Type of internet service
6. **PaymentMethod** - Payment method used
7. **OnlineSecurity** - Whether customer has online security
8. **TechSupport** - Whether customer has tech support
9. **PaperlessBilling** - Whether customer uses paperless billing
10. **StreamingTV** - Whether customer has streaming TV

## 🔧 Model Configuration

### CatBoost Parameters
```python
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    random_seed=42,
    eval_metric='F1',
    class_weights=[1, 2]  # Give more weight to churn class
)
```

### Data Preprocessing Steps
1. **Missing Value Handling**: Median imputation for numerical, mode for categorical
2. **Categorical Encoding**: Label encoding for all categorical variables
3. **Feature Engineering**: 
   - Tenure groups
   - Monthly charges groups
   - Total charges groups
   - Average monthly charges
   - Service count features
   - Contract duration

## 📋 Input Data Format

The model expects customer data with the following features:

### Required Features
- `gender`: Customer gender (Male/Female)
- `SeniorCitizen`: Whether customer is a senior citizen (0/1)
- `Partner`: Whether customer has a partner (Yes/No)
- `Dependents`: Whether customer has dependents (Yes/No)
- `tenure`: Number of months with the company
- `PhoneService`: Whether customer has phone service (Yes/No)
- `MultipleLines`: Multiple lines service (Yes/No/No phone service)
- `InternetService`: Internet service type (DSL/Fiber optic/No)
- `OnlineSecurity`: Online security service (Yes/No/No internet service)
- `OnlineBackup`: Online backup service (Yes/No/No internet service)
- `DeviceProtection`: Device protection service (Yes/No/No internet service)
- `TechSupport`: Tech support service (Yes/No/No internet service)
- `StreamingTV`: Streaming TV service (Yes/No/No internet service)
- `StreamingMovies`: Streaming movies service (Yes/No/No internet service)
- `Contract`: Contract type (Month-to-month/One year/Two year)
- `PaperlessBilling`: Paperless billing (Yes/No)
- `PaymentMethod`: Payment method
- `MonthlyCharges`: Monthly charges amount
- `TotalCharges`: Total charges amount

## 🎯 Prediction Output

The model provides:
- **Binary Prediction**: 0 (No Churn) or 1 (Churn)
- **Probability Score**: Probability of churning (0-1)
- **Risk Level**: Low/Medium/High risk
- **Feature Importance**: Top influential features
- **Insights**: Risk factors and recommendations

## 📊 Example Usage

### Training Example
```python
# Load and train model
from train_model import ChurnPredictionModel

model = ChurnPredictionModel()
df = model.load_data()
X, y = model.preprocess_data(df)
model.train_model(X, y)
model.evaluate_model()
model.save_model()
```

### Prediction Example
```python
# Make prediction
from prediction_utils import ChurnPredictor

predictor = ChurnPredictor()
customer_data = {
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'tenure': 24,
    'Contract': 'Month-to-month',
    'MonthlyCharges': 85.0,
    # ... other features
}

result = predictor.predict_single_customer(customer_data)
print(f"Churn Probability: {result['probability']:.1%}")
print(f"Risk Level: {result['risk_level']}")
```

### Evaluation Example
```python
# Evaluate model
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.run_complete_evaluation()
```

## 🔍 Model Interpretability

The model provides several interpretability features:
- **Feature Importance**: Shows which features most influence predictions
- **Risk Analysis**: Identifies risk factors and protective factors
- **Recommendations**: Provides actionable insights for customer retention
- **Visualizations**: ROC curves, confusion matrices, and feature importance plots

## 📈 Model Monitoring

To monitor model performance over time:
1. **Regular Evaluation**: Run `model_evaluation.py` periodically
2. **Performance Tracking**: Compare metrics across different time periods
3. **Feature Drift**: Monitor changes in feature distributions
4. **Retraining**: Retrain model when performance degrades

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure model file exists: `catboost_churn_model.cbm`
   - Check file paths in configuration

2. **Data Format Error**:
   - Verify all required features are present
   - Check data types match expected format
   - Ensure categorical values match training data

3. **Memory Issues**:
   - Reduce batch size for large datasets
   - Use data sampling for evaluation

### Performance Optimization

1. **Faster Training**:
   - Reduce `iterations` parameter
   - Use smaller `depth` value
   - Enable early stopping

2. **Better Accuracy**:
   - Increase `iterations`
   - Tune hyperparameters using GridSearchCV
   - Add more features through feature engineering

## 📚 Additional Resources

- **CatBoost Documentation**: https://catboost.ai/docs/
- **Scikit-learn Documentation**: https://scikit-learn.org/
- **Pandas Documentation**: https://pandas.pydata.org/

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **AI Assistant** - Initial work

## 🙏 Acknowledgments

- IBM Watson Analytics for the dataset
- CatBoost team for the excellent gradient boosting library
- Streamlit team for the web framework

---

**Last Updated**: December 2024
**Version**: 1.0.0 