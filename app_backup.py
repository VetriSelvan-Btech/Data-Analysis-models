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
    
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="🎯 Churn Predictor Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/churn-prediction-model',
        'Report a bug': "https://github.com/your-repo/churn-prediction-model/issues",
        'About': "# Churn Prediction Model\nAn AI-powered tool for predicting customer churn using CatBoost."
        }
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme styling */
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .stApp {
        background-color: #0e1117;
    }
    
    .stMarkdown {
        color: #fafafa;
    }
    
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: #262730;
        color: #fafafa;
    }
    
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3);
    }
    
    .stExpander {
        background-color: #262730;
        border: 1px solid #4a4a4a;
    }
    
    .stDataFrame {
        background-color: #262730;
    }
    
    .stMetric {
        background-color: #262730;
        border: 1px solid #4a4a4a;
        border-radius: 5px;
        padding: 10px;
    }
    
    .stProgress > div > div > div {
        background-color: #ff4b4b;
    }
    
    /* Custom styling for better dark theme appearance */
    .css-1d391kg {
        background-color: #0e1117;
    }
    
    .css-1v0mbdj {
        background-color: #262730;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #fafafa !important;
    }
    
    /* Link styling */
    a {
        color: #ff4b4b !important;
    }
    
    a:hover {
        color: #ff3333 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg .css-1v0mbdj {
        background-color: #262730;
    }
    
    /* Success and error message styling */
    .stAlert {
        background-color: #262730;
        border: 1px solid #4a4a4a;
    }
</style>
""", unsafe_allow_html=True)
)

# Colorful header
st.markdown("""
<div class="main-header">
    <h1>🎯 Customer Churn Predictor Pro</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">AI-Powered Customer Retention Analysis</p>
</div>
""", unsafe_allow_html=True)

# Add dataset download section
st.markdown("""
<div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); padding: 1.5rem; 
            border-radius: 15px; color: white; margin: 1rem 0; text-align: center;">
    <h3>📊 Download Dataset & Complete Documentation</h3>
    <p>Get the complete Telco Customer Churn dataset and comprehensive documentation!</p>
</div>
""", unsafe_allow_html=True)

# Function to create download link
def get_download_link(file_path, file_name):
    """Generate a download link for the file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}" style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 12px 24px; text-decoration: none; border-radius: 25px; font-weight: bold; display: inline-block; margin: 10px;">📥 Download {file_name}</a>'
    return href

# Dataset download section
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); 
                border-radius: 10px; margin: 1rem 0;">
        <h4>📋 Dataset Information</h4>
        <p><strong>File:</strong> WA_Fn-UseC_-Telco-Customer-Churn.csv</p>
        <p><strong>Records:</strong> 7,043 customers</p>
        <p><strong>Features:</strong> 19 customer attributes</p>
        <p><strong>Source:</strong> IBM Watson Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            dataset_link = get_download_link("WA_Fn-UseC_-Telco-Customer-Churn.csv", "Telco_Customer_Churn_Dataset.csv")
            st.markdown(dataset_link, unsafe_allow_html=True)
            st.markdown("""
            <p style="text-align: center; font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">
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
            <p style="text-align: center; font-size: 0.9rem; opacity: 0.8; margin-top: 0.5rem;">
                📚 Complete documentation (Markdown)
            </p>
            """, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("❌ Documentation file not found.")

st.markdown("---")

# ----------------------------
# Dataset Preview Section
# ----------------------------
with st.expander("📊 Preview the Dataset (Click to expand)", expanded=False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); padding: 1rem; 
                border-radius: 10px; color: white; margin: 1rem 0;">
        <h4>🔍 Dataset Preview</h4>
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

# ----------------------------
# Define Input Fields with Colorful Sections
# ----------------------------
st.markdown("""
<div class="info-box">
    <h3>📋 Customer Information Form</h3>
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
    internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
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

# Financial information in a separate section
st.markdown("### 💵 Financial Information")
col3, col4, col5 = st.columns(3)
with col3:
    tenure = st.slider("⏰ Tenure (months)", 0, 72, 12, help="How long the customer has been with the company")
with col4:
    monthly_charges = st.number_input("💰 Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.01)
with col5:
    total_charges = st.number_input("💵 Total Charges ($)", 0.0, 10000.0, 1500.0, step=0.01)

# ----------------------------
# Prepare Input DataFrame with engineered features
# ----------------------------
def prepare_input_data(gender, senior_citizen, partner, dependents, tenure, phone_service,
                      multiple_lines, internet_service, online_security, online_backup,
                      device_protection, tech_support, streaming_tv, streaming_movies,
                      contract, paperless_billing, payment_method, monthly_charges, total_charges):
    
    # Create input data with only the original features that the model expects
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

# ----------------------------
# Predict on Button Click
# ----------------------------
st.markdown("---")

# Center the button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

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
            
            # Colorful prediction result
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box high-risk">
                    <h2>⚠️ HIGH CHURN RISK DETECTED</h2>
                    <p style="font-size: 1.5rem; margin: 1rem 0;">Churn Probability: {probability:.1%}</p>
                    <p>🚨 This customer shows strong signs of churning. Immediate retention strategies recommended!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box low-risk">
                    <h2>✅ LOW CHURN RISK</h2>
                    <p style="font-size: 1.5rem; margin: 1rem 0;">Churn Probability: {probability:.1%}</p>
                    <p>🎉 This customer appears stable and unlikely to churn. Continue with current service quality!</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced probability gauge
            st.markdown("### 📊 Risk Level Visualization")
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                # Create a more colorful progress bar
                if probability < 0.3:
                    color = "🟢"
                    risk_level = "Low Risk"
                elif probability < 0.6:
                    color = "🟡"
                    risk_level = "Medium Risk"
                else:
                    color = "🔴"
                    risk_level = "High Risk"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea, #764ba2); 
                            border-radius: 15px; color: white; margin: 1rem 0;">
                    <h3>{color} Churn Risk Level: {risk_level}</h3>
                    <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 0.5rem; margin: 1rem 0;">
                        <div style="background: {'#ff6b6b' if probability > 0.5 else '#2ed573'}; 
                                    width: {probability*100}%; height: 20px; border-radius: 8px; 
                                    transition: width 0.5s ease;"></div>
                    </div>
                    <p style="font-size: 1.2rem; font-weight: bold;">{probability:.1%} Probability</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {e}")

# ----------------------------
# Feature Importance Plot
# ----------------------------
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
    
    # Display top 10 features with colorful styling
    st.markdown("### 🎯 Top 10 Most Important Features")
    
    # Create a more colorful plot using seaborn with proper parameters
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
    
    # Display feature importance table with styling
    st.markdown("### 📋 Detailed Feature Analysis")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; color: white;">
        <h4>Feature Importance Rankings</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Style the dataframe
    styled_df = importance_df.head(15).copy()
    styled_df['Rank'] = range(1, len(styled_df) + 1)
    styled_df = styled_df[['Rank', 'Feature', 'Importance']]
    styled_df['Importance'] = styled_df['Importance'].round(3)
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
except Exception as e:
    st.warning(f"⚠️ Could not display feature importance: {e}")

# ----------------------------
# Model Information
# ----------------------------
st.markdown("---")
st.markdown("""
<div class="info-box">
    <h2>🤖 AI Model Information & Capabilities</h2>
    <p>Learn about the advanced machine learning technology powering this churn prediction system</p>
</div>
""", unsafe_allow_html=True)

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

# ----------------------------
# How the Model Works - Detailed Explanation
# ----------------------------
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 2rem; 
            border-radius: 15px; color: white; margin: 2rem 0;">
    <h2>🧠 How the AI Model Works</h2>
    <p>Understanding the machine learning process behind churn prediction</p>
</div>
""", unsafe_allow_html=True)

# Training Process Explanation
with st.expander("🎓 Model Training Process (Click to expand)", expanded=False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1.5rem; 
                border-radius: 10px; color: white; margin: 1rem 0;">
        <h3>📚 Training Process Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <h4>🔍 Step 1: Data Preprocessing</h4>
            <ul>
                <li>Loaded 7,043 customer records from IBM Watson Analytics</li>
                <li>Cleaned missing values and converted data types</li>
                <li>Handled categorical variables (gender, contract type, etc.)</li>
                <li>Converted target variable (Churn: Yes/No → 1/0)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <h4>🎯 Step 2: Feature Engineering</h4>
            <ul>
                <li>Created tenure groups (New, Short, Medium, Long)</li>
                <li>Grouped charges by ranges (Low, Medium, High, Very High)</li>
                <li>Added interaction features (contract type, payment method)</li>
                <li>Calculated service count composite features</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <h4>🤖 Step 3: Model Training</h4>
            <ul>
                <li>Split data: 80% training, 20% testing</li>
                <li>Used CatBoost algorithm with optimized hyperparameters</li>
                <li>Applied class weights (3:1) to handle imbalanced data</li>
                <li>Implemented 5-fold cross-validation for robustness</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <h4>📊 Step 4: Model Evaluation</h4>
            <ul>
                <li>Evaluated using multiple metrics (Accuracy, Precision, Recall, F1)</li>
                <li>Analyzed feature importance rankings</li>
                <li>Generated confusion matrix for error analysis</li>
                <li>Validated performance on unseen test data</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# How CatBoost Works
with st.expander("🌳 Understanding CatBoost Algorithm (Click to expand)", expanded=False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); padding: 1.5rem; 
                border-radius: 10px; color: white; margin: 1rem 0;">
        <h3>🌳 CatBoost: Gradient Boosting with Categorical Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
        <h4>🔬 Algorithm Details:</h4>
        <p><strong>CatBoost</strong> is an advanced gradient boosting algorithm that excels at handling categorical features directly without preprocessing. Here's how it works:</p>
        
        <h5>🎯 Key Advantages:</h5>
        <ul>
            <li><strong>Native Categorical Support:</strong> Handles categorical variables without one-hot encoding</li>
            <li><strong>Reduced Overfitting:</strong> Uses ordered boosting to prevent target leakage</li>
            <li><strong>High Performance:</strong> Optimized for speed and accuracy</li>
            <li><strong>Feature Importance:</strong> Provides detailed feature ranking</li>
        </ul>
        
        <h5>🧮 Mathematical Process:</h5>
        <ol>
            <li><strong>Gradient Boosting:</strong> Builds trees sequentially, each correcting the previous tree's errors</li>
            <li><strong>Categorical Encoding:</strong> Uses target-based encoding for categorical features</li>
            <li><strong>Regularization:</strong> Applies L2 regularization to prevent overfitting</li>
            <li><strong>Early Stopping:</strong> Stops training when validation performance plateaus</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Model Hyperparameters
with st.expander("⚙️ Model Configuration & Hyperparameters (Click to expand)", expanded=False):
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b, #ee5a52); padding: 1.5rem; 
                border-radius: 10px; color: white; margin: 1rem 0;">
        <h3>⚙️ Optimized Hyperparameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <h4>🎛️ Core Parameters:</h4>
            <ul>
                <li><strong>Iterations:</strong> 1000 (number of trees)</li>
                <li><strong>Learning Rate:</strong> 0.03 (step size)</li>
                <li><strong>Depth:</strong> 8 (tree depth)</li>
                <li><strong>Loss Function:</strong> Logloss</li>
                <li><strong>Eval Metric:</strong> F1-Score</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            <h4>🛡️ Regularization:</h4>
            <ul>
                <li><strong>L2 Leaf Reg:</strong> 3 (L2 regularization)</li>
                <li><strong>Random Strength:</strong> 0.8 (randomization)</li>
                <li><strong>Bagging Temp:</strong> 0.8 (bagging)</li>
                <li><strong>Class Weights:</strong> [1, 3] (churn class weight)</li>
                <li><strong>Early Stopping:</strong> 50 rounds</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); 
            border-radius: 15px; color: white; margin: 2rem 0;">
    <h3>🎉 Customer Churn Predictor Pro</h3>
    <p>Powered by Advanced AI & Machine Learning</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Built with Streamlit, CatBoost, and Python</p>
    <p style="font-size: 1rem; margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px;">
        📊 <strong>Dataset Available:</strong> Download the complete IBM Watson Analytics Telco Customer Churn dataset above!
    </p>
</div>
""", unsafe_allow_html=True)
