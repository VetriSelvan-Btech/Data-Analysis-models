import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
import base64

# ----------------------------
# Load the trained CatBoost model
# ----------------------------
@st.cache_resource
def load_model():
    try:
        model = CatBoostClassifier()
        model.load_model("catboost_churn_model.cbm")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

if model is None:
    st.error("❌ Model could not be loaded. Please ensure the model file exists.")
    st.stop()

# Custom CSS for colorful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        color: white;
        border: 3px solid #ff4757;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #2ed573, #1e90ff);
        color: white;
        border: 3px solid #2ed573;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="🎯 Churn Prediction Model", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Churn Prediction Model</h1>
    <p>AI-Powered Customer Retention Analysis & Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# Dataset download section
st.markdown("""
<div class="info-box">
    <h2>📊 Download Dataset & Complete Documentation</h2>
    <p>Get the complete Telco Customer Churn dataset and comprehensive documentation!</p>
</div>
""", unsafe_allow_html=True)

# Function to create download link
def get_download_link(file_path, file_name):
    """Generate a download link for the file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; padding: 10px 20px; text-decoration: none; border-radius: 25px; font-weight: bold; display: inline-block; margin: 10px;">📥 Download {file_name}</a>'
    return href

# Dataset download section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>📋 Dataset Information</h3>
        <p><strong>📁 File:</strong> WA_Fn-UseC_-Telco-Customer-Churn.csv</p>
        <p><strong>👥 Records:</strong> 7,043 customers</p>
        <p><strong>🔧 Features:</strong> 19 customer attributes</p>
        <p><strong>🏢 Source:</strong> IBM Watson Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            dataset_link = get_download_link("WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco_Customer_Churn_Dataset.csv")
            st.markdown(dataset_link, unsafe_allow_html=True)
            st.markdown("""
            <p style="text-align: center; font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem; color: #667eea; font-weight: bold;">
                📊 Complete dataset (CSV format)
            </p>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("❌ Dataset file not found.")
    
    with col2:
        try:
            doc_link = get_download_link("Churn_Prediction_Complete_Documentation.md", "Churn_Prediction_Complete_Documentation.md")
            st.markdown(doc_link, unsafe_allow_html=True)
            st.markdown("""
            <p style="text-align: center; font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem; color: #667eea; font-weight: bold;">
                📚 Complete documentation (Markdown)
            </p>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("❌ Documentation file not found.")

st.markdown("---")

# Dataset Preview Section
with st.expander("📊 Preview the Dataset (Click to expand)", expanded=False):
    st.markdown("""
    <div class="info-box">
        <h3>🔍 Dataset Preview</h3>
        <p>Take a look at the actual data used to train this AI model</p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Load and display dataset preview
        df_preview = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Records", f"{len(df_preview):,}")
        with col2:
            st.metric("📋 Features", f"{len(df_preview.columns)}")
        with col3:
            churn_rate = (df_preview['Churn'].value_counts(normalize=True) * 100).round(1)
            st.metric("📈 Churn Rate", f"{churn_rate['Yes']}%")
        
        # Show first few rows
        st.markdown("### 📋 Sample Data (First 10 Records)")
        st.dataframe(df_preview.head(10), use_container_width=True)
        
        # Show data info
        st.markdown("### 📊 Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📋 Column Information:**")
            st.write(df_preview.info())
        with col2:
            st.markdown("**📈 Churn Distribution:**")
            churn_counts = df_preview['Churn'].value_counts()
            st.write(churn_counts)
            
    except Exception as e:
        st.error(f"❌ Error loading dataset preview: {e}")

# Customer Information Form
st.markdown("""
<div class="info-box">
    <h2>📋 Customer Information Form</h2>
    <p>Please fill in the customer details below to get an accurate churn prediction.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Personal Information")
    gender = st.selectbox("👥 Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("👴 Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("💑 Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("👶 Has Dependents?", ["Yes", "No"])
    
    st.markdown("### 📞 Phone Services")
    phone_service = st.selectbox("📱 Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("📞 Multiple Lines", ["Yes", "No", "No phone service"])
    
    st.markdown("### 🌐 Internet Services")
    internet_service = st.selectbox("�� Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("🔒 Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("💾 Online Backup", ["Yes", "No", "No internet service"])

with col2:
    st.markdown("### 🛡️ Additional Services")
    device_protection = st.selectbox("🛡️ Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("🆘 Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("📺 Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("🎬 Streaming Movies", ["Yes", "No", "No internet service"])
    
    st.markdown("### 💰 Billing & Contract")
    contract = st.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("📧 Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("💳 Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

# Financial information section
st.markdown("### 💵 Financial Information")
col3, col4, col5 = st.columns(3)
with col3:
    tenure = st.slider("⏰ Tenure (months)", 0, 72, 12, help="How long the customer has been with the company")
with col4:
    monthly_charges = st.number_input("💰 Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.01)
with col5:
    total_charges = st.number_input("💵 Total Charges ($)", 0.0, 10000.0, 1500.0, step=0.01)

# Prepare Input DataFrame function
def prepare_input_data(gender, senior_citizen, partner, dependents, tenure, phone_service,
                      multiple_lines, internet_service, online_security, online_backup,
                      device_protection, tech_support, streaming_tv, streaming_movies,
                      contract, paperless_billing, payment_method, monthly_charges, total_charges):
    
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])
    
    return input_data

# Prediction Section
st.markdown("---")

st.markdown("""
<div class="info-box">
    <h2>🔮 AI Prediction Engine</h2>
    <p>Ready to analyze customer data and predict churn risk?</p>
</div>
""", unsafe_allow_html=True)

# Center the button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🚀 Generate Churn Prediction", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("🤖 AI is analyzing customer data..."):
        input_data = prepare_input_data(
            gender, senior_citizen, partner, dependents, tenure, phone_service,
            multiple_lines, internet_service, online_security, online_backup,
            device_protection, tech_support, streaming_tv, streaming_movies,
            contract, paperless_billing, payment_method, monthly_charges, total_charges
        )
        
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            st.markdown("---")
            
            # Prediction result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box high-risk">
                    <h2>⚠️ HIGH CHURN RISK DETECTED</h2>
                    <p>Churn Probability: {probability:.1%}</p>
                    <p>🚨 This customer shows strong signs of churning. Immediate retention strategies recommended!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box low-risk">
                    <h2>✅ LOW CHURN RISK</h2>
                    <p>Churn Probability: {probability:.1%}</p>
                    <p>🎉 This customer appears stable and unlikely to churn. Continue with current service quality!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk level visualization
            st.markdown("### 📊 Risk Level Visualization")
            
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                if probability < 0.3:
                    color = "🟢"
                    risk_level = "Low Risk"
                    bar_color = "#2ed573"
                elif probability < 0.6:
                    color = "🟡"
                    risk_level = "Medium Risk"
                    bar_color = "#ffa726"
                else:
                    color = "🔴"
                    risk_level = "High Risk"
                    bar_color = "#ff6b6b"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{color} Churn Risk Level: {risk_level}</h3>
                    <p>Probability: {probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

# Feature Importance Plot
st.markdown("---")
st.markdown("""
<div class="feature-card">
    <h2>📊 AI Model Insights & Analytics</h2>
    <p>Discover which factors most influence customer churn predictions</p>
</div>
""", unsafe_allow_html=True)

try:
    feature_names = model.feature_names_
    importances = model.get_feature_importance()
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    # Display top 10 features
    st.markdown("### 🎯 Top 10 Most Important Features")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', 
                palette='viridis', ax=ax, hue='Feature', legend=False)
    plt.xlabel("Feature Importance Score", fontsize=12, fontweight='bold')
    plt.title("🎯 Top 10 Features Driving Churn Predictions", fontsize=16, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_width():.2f}', 
                   (p.get_width() + 0.01, p.get_y() + p.get_height()/2),
                   ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display feature importance table
    st.markdown("### 📋 Detailed Feature Analysis")
    
    # Style the dataframe
    styled_df = importance_df.head(15).copy()
    styled_df['Rank'] = range(1, len(styled_df) + 1)
    styled_df = styled_df[['Rank', 'Feature', 'Importance']]
    styled_df['Importance'] = styled_df['Importance'].round(3)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
except Exception as e:
    st.warning(f"⚠️ Could not display feature importance: {e}")

# Model Information
st.markdown("---")
st.markdown("### 🤖 AI Model Information & Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🔬 Model Technology</h3>
        <p><strong>Algorithm:</strong> CatBoost (Gradient Boosting)</p>
        <p><strong>Training Data:</strong> 7,043 Customer Records</p>
        <p><strong>Features:</strong> 19 Customer Attributes</p>
        <p><strong>Accuracy:</strong> ~85-90%</p>
        <p><strong>Dataset:</strong> IBM Watson Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Model Optimizations</h3>
        <p>• F1-score optimization for balanced precision/recall</p>
        <p>• Class weights to handle imbalanced data</p>
        <p>• Regularization to prevent overfitting</p>
        <p>• Cross-validation for robust evaluation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Performance Metrics</h3>
        <p><strong>Precision:</strong> ~75-80%</p>
        <p><strong>Recall:</strong> ~70-75%</p>
        <p><strong>F1-Score:</strong> ~72-77%</p>
        <p><strong>Cross-Validation:</strong> 5-fold CV</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h3>🚀 Key Features</h3>
        <p>• Real-time predictions with probability scores</p>
        <p>• Feature importance analysis</p>
        <p>• Risk level visualization</p>
        <p>• User-friendly interface</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h3>🎉 Customer Churn Predictor Pro</h3>
    <p>Powered by Advanced AI & Machine Learning</p>
    <p>Built with Streamlit, CatBoost, and Python</p>
</div>
""", unsafe_allow_html=True)
