# 🎯 Churn Prediction Model

An advanced machine learning model for predicting customer churn using CatBoost with enhanced features, improved precision/accuracy, and a beautiful web interface for real-time predictions.

## 🚀 Features

- **Improved Model Performance**: Enhanced CatBoost model with better hyperparameters
- **Feature Engineering**: Advanced feature creation for better prediction accuracy
- **Comprehensive Evaluation**: Cross-validation, confusion matrix, and detailed metrics
- **User-Friendly Interface**: Beautiful Streamlit web application
- **Real-time Predictions**: Instant churn risk assessment with probability scores

## 📊 Model Improvements

### Enhanced Features
- **Tenure Groups**: Categorized customer tenure (New, Short, Medium, Long)
- **Charge Groups**: Grouped monthly and total charges for better patterns
- **Interaction Features**: Contract type, payment method, and internet service interactions
- **Composite Features**: Service count to capture customer engagement

### Model Optimizations
- **F1-Score Focus**: Balanced precision and recall optimization
- **Class Weights**: Handles imbalanced churn data (3:1 weight for churn class)
- **Regularization**: L2 regularization to prevent overfitting
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Early Stopping**: Prevents overfitting during training

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the dataset file**:
   - `WA_Fn-UseC_-Telco-Customer-Churn.csv` should be in the project directory

## 🎯 Usage

### Step 1: Train the Model
```bash
python train_model.py
```
This will:
- Load and preprocess the data
- Train the improved CatBoost model
- Generate evaluation metrics and visualizations
- Save the model as `catboost_churn_model.cbm`

### Step 2: Run the Web Application
```bash
streamlit run app.py
```
This will:
- Start the Streamlit web server
- Open the churn prediction interface in your browser
- Allow you to input customer data and get predictions

## 📈 Model Performance

The improved model typically achieves:
- **Accuracy**: ~85-90%
- **Precision**: ~75-80%
- **Recall**: ~70-75%
- **F1-Score**: ~72-77%

*Note: Actual performance may vary based on the dataset and training conditions.*

## 🎨 Web Application Features

- **Two-Column Layout**: Organized input fields for better UX
- **Real-time Predictions**: Instant churn risk assessment
- **Probability Display**: Visual progress bar showing churn probability
- **Feature Importance**: Interactive charts showing most important features
- **Model Insights**: Detailed information about the model's capabilities

## 📁 Project Structure

```
churn prediction model/
├── app.py                          # Streamlit web application
├── model.py                        # Improved model training script
├── train_model.py                  # Simple training runner
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── catboost_churn_model.cbm        # Trained model (generated)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
├── confusion_matrix.png            # Generated evaluation plot
└── feature_importance.png          # Generated feature importance plot
```

## 🔧 Customization

### Model Parameters
You can modify the model hyperparameters in `model.py`:
- `iterations`: Number of boosting iterations
- `learning_rate`: Learning rate for gradient boosting
- `depth`: Maximum depth of trees
- `class_weights`: Weights for handling class imbalance

### Feature Engineering
Add new features in the `load_and_preprocess_data()` function:
- Create new categorical groups
- Add interaction features
- Implement domain-specific features

## 🐛 Troubleshooting

### Common Issues

1. **Model not loading**: Ensure `catboost_churn_model.cbm` exists
2. **Missing dataset**: Check if `WA_Fn-UseC_-Telco-Customer-Churn.csv` is present
3. **Package errors**: Run `pip install -r requirements.txt`
4. **Memory issues**: Reduce `iterations` parameter in model training

### Error Messages
- **"Model could not be loaded"**: Train the model first using `python train_model.py`
- **"Dataset not found"**: Ensure the CSV file is in the correct location
- **"Package not found"**: Install missing packages with pip

## 📝 License

This project is for educational and demonstration purposes.

## 🤝 Contributing

Feel free to improve the model by:
- Adding new features
- Optimizing hyperparameters
- Enhancing the web interface
- Improving documentation

---

**Happy Churn Prediction! 🎯** 