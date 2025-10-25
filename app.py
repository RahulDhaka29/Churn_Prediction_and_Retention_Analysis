import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Telco Churn Prediction | By Rahul",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CUSTOM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Animated gradient background */
.main .block-container {
    background: linear-gradient(-45deg, #f5f7fa, #e8ecf1, #f0f2f6, #e3e7ed);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Enhanced Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    border-right: none;
    box-shadow: 4px 0 20px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stSidebar"] h1 {
    font-weight: 800;
    color: #ffffff !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    margin-bottom: 1.5rem;
}

[data-testid="stSidebar"] .stMarkdown {
    color: #ffffff !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #ffffff !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] label {
    color: #ffffff !important;
}

[data-testid="stSidebar"] p {
    color: #ffffff !important;
}

[data-testid="stSidebar"] span {
    color: #ffffff !important;
}

[data-testid="stSidebar"] div {
    color: #ffffff !important;
}

/* Sidebar navigation (radio buttons) with hover animations */
div[role="radiogroup"] label {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: block;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    backdrop-filter: blur(10px);
}

div[role="radiogroup"] label:hover {
    background: rgba(255, 75, 75, 0.2);
    border-color: #FF4B4B;
    color: #ffffff !important;
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
}

/* Selected radio button with gradient */
div[role="radiogroup"] [aria-checked="true"] {
    background: linear-gradient(135deg, #FF4B4B 0%, #ff6b6b 100%) !important;
    color: #FFFFFF !important;
    border-color: #FF4B4B !important;
    transform: translateX(5px);
    box-shadow: 0 6px 20px rgba(255, 75, 75, 0.5);
}

div[role="radiogroup"] [aria-checked="true"]:hover {
    background: linear-gradient(135deg, #E03C3C 0%, #ff5252 100%) !important;
    box-shadow: 0 8px 25px rgba(255, 75, 75, 0.6);
}

/* Force white text on radio buttons */
div[role="radiogroup"] label div {
    color: #ffffff !important;
}

div[role="radiogroup"] [aria-checked="true"] div {
    color: #ffffff !important;
}

/* Glass-morphism cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    padding: 2.5rem;
    margin-bottom: 2rem;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.12);
    transform: translateY(-5px);
    border-color: rgba(255, 75, 75, 0.3);
}

/* Enhanced stat cards with gradients */
.stat-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border: none;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    height: 100%;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #FF4B4B, #ff6b6b, #FF4B4B);
    background-size: 200% 100%;
    animation: shimmer 3s linear infinite;
}

@keyframes shimmer {
    0% { background-position: -100% 0; }
    100% { background-position: 100% 0; }
}

.stat-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 40px rgba(255, 75, 75, 0.15);
}

.stat-value {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF4B4B, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.stat-label {
    font-size: 1.1rem;
    font-weight: 600;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Modern headers with underline animation */
h1, h2, h3 {
    color: #1a1a2e;
    font-weight: 800;
    position: relative;
    padding-bottom: 1rem;
}

h1 {
    font-size: 2.5rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.5rem;
}

h1::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    width: 80px;
    height: 4px;
    background: linear-gradient(90deg, #FF4B4B, #ff6b6b);
    border-radius: 2px;
}

h2 {
    font-size: 1.8rem;
    color: #16213e;
}

/* Enhanced tabs */
[data-testid="stTabs"] button[role="tab"] {
    font-size: 1.05rem;
    font-weight: 600;
    color: #666;
    padding: 1rem 2rem;
    border-radius: 12px 12px 0 0;
    transition: all 0.3s ease;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: #FF4B4B;
    background: rgba(255, 75, 75, 0.05);
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #FF4B4B;
    background: rgba(255, 75, 75, 0.1);
    border-bottom: 3px solid #FF4B4B;
    font-weight: 700;
}

/* Enhanced buttons */
.stButton > button {
    background: linear-gradient(135deg, #FF4B4B 0%, #ff6b6b 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 700;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #E03C3C 0%, #ff5252 100%);
    box-shadow: 0 6px 25px rgba(255, 75, 75, 0.4);
    transform: translateY(-2px);
}

/* Enhanced link buttons */
.stLinkButton > a {
    background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    text-decoration: none;
    display: inline-block;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(22, 33, 62, 0.3);
}

.stLinkButton > a:hover {
    background: linear-gradient(135deg, #FF4B4B 0%, #ff6b6b 100%);
    box-shadow: 0 6px 25px rgba(255, 75, 75, 0.4);
    transform: translateY(-2px);
}

/* Expander styling */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(255, 75, 75, 0.2);
    border-radius: 12px;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

[data-testid="stExpander"] summary {
    font-size: 0 !important;
    font-weight: 700;
    padding: 1rem 1.5rem;
    color: #16213e;
}

[data-testid="stExpander"] summary span {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: #16213e !important;
}

[data-testid="stExpander"] summary svg {
    visibility: hidden !important;
    display: none !important;
}

[data-testid="stExpander"] summary::before {
    content: '▶';
    font-size: 1rem !important;
    color: #FF4B4B;
    transition: transform 0.3s ease;
    display: inline-block;
    margin-right: 0.75rem;
}

[data-testid="stExpander"][aria-expanded="true"] summary::before {
    transform: rotate(90deg);
}

/* Sidebar collapse button */
[data-testid="stSidebarCollapseButton"] {
    color: #FF4B4B;
    font-size: 0 !important;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    transition: all 0.3s ease;
}

[data-testid="stSidebarCollapseButton"]:hover {
    background: rgba(255, 75, 75, 0.2);
}

[data-testid="stSidebarCollapseButton"] svg {
    visibility: hidden !important;
    display: none !important;
}

[data-testid="stSidebarCollapseButton"]::before {
    content: '«';
    font-size: 1.5rem !important;
    font-weight: bold;
}

/* Info boxes with icons */
.stInfo, .stSuccess, .stWarning, .stError {
    border-radius: 12px;
    border: none;
    padding: 1.25rem;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}

/* Enhanced dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}

/* Input fields styling */
.stSelectbox, .stSlider {
    margin-bottom: 1.5rem;
}

.stSelectbox > label, .stSlider > label {
    font-weight: 600;
    color: #16213e;
    margin-bottom: 0.5rem;
}

/* Enhanced footer */
.footer {
    position: relative;
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    background: linear-gradient(135deg, rgba(26, 26, 46, 0.05), rgba(22, 33, 62, 0.05));
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

.footer p {
    color: #666;
    font-size: 1rem;
    font-weight: 500;
    margin: 0;
}

.footer a {
    color: #FF4B4B;
    text-decoration: none;
    font-weight: 700;
    transition: all 0.3s ease;
    position: relative;
}

.footer a::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 0;
    height: 2px;
    background: #FF4B4B;
    transition: width 0.3s ease;
}

.footer a:hover::after {
    width: 100%;
}

.footer a:hover {
    color: #E03C3C;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    h1 {
        font-size: 2rem;
    }
    
    .stat-value {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
def load_preprocessor_direct():
    print("Attempting to load preprocessor.pkl directly...")
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    print("Preprocessor loaded successfully!")
    return preprocessor

def load_model_direct():
    print("Attempting to load model.pkl directly...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
    return model

@st.cache_data
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Load assets
try:
    preprocessor = load_preprocessor_direct()
    model = load_model_direct()
    results_df = load_csv('model_comparison_results.csv')
    raw_df = load_csv('Telco-Customer-Churn.csv')
except FileNotFoundError as e:
    st.error(f"Error: Missing file! Make sure the following files are in the same folder as app.py:")
    st.error(f"- {e.filename}")
    st.error("- preprocessor.pkl")
    st.error("- model.pkl")
    st.error("- model_comparison_results.csv")
    st.error("- Telco-Customer-Churn.csv")
    st.stop()
except Exception as e:
    st.error(f"FATAL ERROR loading .pkl files: {e}")
    st.error("This strongly suggests a version mismatch or corrupted file.")
    st.error("Ensure you restarted the notebook kernel and re-ran 'Run All' to save fresh .pkl files.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80", use_container_width=True)
    st.title("🎯 Churn Analytics")
    
    user_menu = st.radio(
        'Navigate',
        ('🏠 Project Overview', 
         '📊 Retention Analysis', 
         '🤖 Live Predictor')
    )
    st.markdown("---")
    st.info("💡 AI-powered churn prediction using machine learning to help retain valuable customers.")

# --- PAGE 1: PROJECT OVERVIEW ---
if user_menu == '🏠 Project Overview':
    st.title("🚀 Customer Churn Prediction & Analytics")
    st.markdown("### Transform customer data into actionable retention strategies")
    
    with st.container(border=True):
        st.header("🎯 The Business Challenge")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            Customer churn represents one of the most critical challenges in business today. 
            Industry research shows it costs **5-25x more** to acquire a new customer than to retain an existing one.
            
            **Our Mission:**
            - 🔍 **Understand WHY** customers leave through deep data analysis
            - 🎯 **Predict WHO** is likely to churn before they leave
            - 💡 **Recommend HOW** to prevent churn with targeted interventions
            """)
        with col2:
            st.markdown("""
            <div class="stat-card">
                <div class="stat-value">5-25x</div>
                <div class="stat-label">Cost Difference</div>
            </div>
            """, unsafe_allow_html=True)

    with st.container(border=True):
        st.header("⚙️ Our Data Science Pipeline")
        
        pipeline_steps = [
            ("📥 Data Loading", "Imported and explored 7,043 customer records with 21 features"),
            ("🧹 Data Cleaning", "Handled missing values, fixed data types, encoded target variable"),
            ("📊 EDA & Analysis", "Discovered key churn patterns and retention factors"),
            ("🔧 Feature Engineering", "Built preprocessing pipeline with scaling and encoding"),
            ("🏆 Model Training", "Compared 6 ML algorithms, optimized for recall"),
            ("💾 Deployment", "Saved best model for real-time predictions")
        ]
        
        for i, (title, desc) in enumerate(pipeline_steps, 1):
            with st.expander(f"**Step {i}: {title}**"):
                st.markdown(f"_{desc}_")

    with st.container(border=True):
        st.header("👨‍💻 About This Project")
        st.markdown("""
        This end-to-end machine learning application was developed by **Rahul** as a portfolio showcase, 
        demonstrating expertise in:
        - 🐍 Python & Data Science Libraries
        - 🤖 Machine Learning & Model Optimization
        - 📊 Data Visualization & Storytelling
        - 🎨 Web Application Development
        """)
        
        st.markdown("### 🔗 Connect & Explore")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.link_button("💻 View Source Code", "https://github.com/RahulDhaka29")
        with col2:
            st.link_button("👔 LinkedIn Profile", "https://www.linkedin.com/in/rahul-dhaka-56b975289/")
        with col3:
            st.link_button("📊 Dataset on Kaggle", "https://www.kaggle.com/datasets/blastchar/telco-customer-churn")

# --- PAGE 2: RETENTION ANALYSIS ---
elif user_menu == '📊 Retention Analysis':
    st.title("📊 Customer Retention Insights")
    st.markdown("### Data-driven analysis of churn patterns and retention factors")
    
    with st.container(border=True):
        st.header("📈 Churn Overview")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            churn_rate = (raw_df['Churn'].value_counts(normalize=True)['Yes'] * 100)
            total_customers = len(raw_df)
            churned_customers = raw_df['Churn'].value_counts()['Yes']
            
            st.markdown(f'<div class="stat-card"><div class="stat-value">{churn_rate:.1f}%</div><div class="stat-label">Churn Rate</div></div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{churned_customers:,}</div><div class="stat-label">Churned</div></div>', unsafe_allow_html=True)
        
        with col3:
            churn_counts = raw_df['Churn'].value_counts().reset_index()
            churn_counts.columns = ['Churn_Status', 'Count']
            fig_pie = px.pie(churn_counts, names='Churn_Status', values='Count',
                             title='Customer Distribution',
                             color_discrete_map={'No': '#0068C9', 'Yes': '#FF4B4B'},
                             hole=0.4)
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12, color='#1a1a2e')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        st.info("📌 Over 1 in 4 customers churned in this dataset. Our ML model helps identify these at-risk customers before they leave.")

    with st.container(border=True):
        st.header("🧠 Key Predictive Features")
        st.markdown("Our model identified the most important factors influencing customer churn:")
        
        try:
            feature_names = preprocessor.get_feature_names_out()
            
            if hasattr(model, 'coef_'):
                importances = model.coef_[0]
            elif hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                st.warning("Could not extract feature importances from this model type.")
                importances = None

            if importances is not None:
                feature_importances_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                if hasattr(model, 'coef_'):
                    churn_drivers = feature_importances_df.head(10).sort_values(by='Importance', ascending=False)
                    churn_protectors = feature_importances_df.tail(10).sort_values(by='Importance', ascending=True)
                    plot_df = pd.concat([churn_drivers, churn_protectors]).sort_values(by='Importance')
                    
                    fig_imp = px.bar(plot_df, x='Importance', y='Feature', orientation='h',
                                 color='Importance',
                                 color_continuous_scale=px.colors.diverging.RdBu,
                                 title='Top Churn Drivers vs Retention Factors')
                    
                    fig_imp.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12, color='#1a1a2e'),
                        height=600
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.error("**🔴 Churn Drivers (Positive Values)**")
                        st.markdown("Features that increase likelihood of churn:")
                        st.markdown("- Month-to-month contracts")
                        st.markdown("- Electronic check payments")
                        st.markdown("- Fiber optic without add-ons")
                    
                    with col2:
                        st.success("**🟢 Retention Factors (Negative Values)**")
                        st.markdown("Features that decrease likelihood of churn:")
                        st.markdown("- Two-year contracts")
                        st.markdown("- Long tenure")
                        st.markdown("- Tech support subscriptions")
                
                else:
                    plot_df = feature_importances_df.head(15)
                    
                    fig_imp = px.bar(plot_df, x='Importance', y='Feature', orientation='h',
                                 color='Importance',
                                 color_continuous_scale=px.colors.sequential.Reds,
                                 title='Top 15 Most Important Features')
                    fig_imp.update_layout(
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(size=12, color='#1a1a2e'),
                        height=600
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    st.info("**ℹ️ Note:** This chart shows feature importance magnitudes. Higher values indicate greater impact on predictions.")

        except Exception as e:
            st.error(f"Could not generate feature importance plot. Error: {e}")

    with st.container(border=True):
        st.header("🏆 Model Performance Comparison")
        st.markdown("We tested 6 different machine learning algorithms to find the best performer:")
        
        try:
            if 'Model' not in results_df.columns and results_df.index.name == 'Model':
                results_df_display = results_df
            elif 'Model' in results_df.columns:
                results_df_display = results_df.set_index('Model')
            else:
                results_df_display = results_df
                st.warning("Could not find 'Model' column for indexing comparison results.")

        except Exception as e:
            st.warning(f"Error setting index for comparison results: {e}")
            results_df_display = results_df

        st.dataframe(results_df_display, use_container_width=True)
        
        try:
            if results_df_display.index.name == 'Model':
                winner_model = results_df_display.index[0]
                winner_recall = results_df_display.iloc[0]['Recall (Churn=1)']
            else:
                winner_model = results_df_display.iloc[0]['Model']
                winner_recall = results_df_display.iloc[0]['Recall (Churn=1)']

            st.success(f"✅ **Winner:** `{winner_model}` achieved the highest Recall score of **{winner_recall:.2f}**, meaning it correctly identifies {winner_recall*100:.0f}% of customers who will churn!")
        except Exception as e:
            st.warning(f"Could not display winner from 'model_comparison_results.csv'. Error: {e}")

# --- PAGE 3: LIVE CHURN PREDICTOR ---
elif user_menu == '🤖 Live Predictor':
    st.title("🤖 Real-Time Churn Prediction")
    st.markdown("### Enter customer details below to get instant churn risk assessment")
    
    with st.container(border=True):
        st.subheader("📋 Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Account & Contract Details**")
            tenure = st.slider("📅 Tenure (Months)", 0, 72, 12, help="How long has the customer been with us?")
            Contract = st.selectbox("📝 Contract Type", ['Month-to-month', 'One year', 'Two year'])
            PaymentMethod = st.selectbox("💳 Payment Method", 
                ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            PaperlessBilling = st.selectbox("📄 Paperless Billing", ['Yes', 'No'])
            MonthlyCharges = st.slider("💰 Monthly Charges ($)", 18.0, 120.0, 70.0, 0.01)
            TotalCharges = tenure * MonthlyCharges
            st.info(f"📊 **Calculated Total Charges:** ${TotalCharges:.2f}")
        
        with col2:
            st.markdown("**Demographics**")
            gender = st.selectbox("👤 Gender", ['Male', 'Female'])
            SeniorCitizen_input = st.selectbox("👴 Senior Citizen", ['No', 'Yes'])
            Partner = st.selectbox("💑 Has Partner", ['No', 'Yes'])
            Dependents = st.selectbox("👨‍👩‍👧‍👦 Has Dependents", ['No', 'Yes'])
            PhoneService = st.selectbox("📞 Phone Service", ['Yes', 'No'])
            MultipleLines = st.selectbox("📱 Multiple Lines", ['No phone service', 'No', 'Yes'])

    with st.container(border=True):
        st.subheader("🌐 Internet & Add-on Services")
        
        cols_services = st.columns(3)
        
        with cols_services[0]:
            InternetService = st.selectbox("🌐 Internet Service", ['DSL', 'Fiber optic', 'No'])
            OnlineSecurity = st.selectbox("🔒 Online Security", ['No internet service', 'No', 'Yes'])
            OnlineBackup = st.selectbox("💾 Online Backup", ['No internet service', 'No', 'Yes'])
        
        with cols_services[1]:
            DeviceProtection = st.selectbox("📱 Device Protection", ['No internet service', 'No', 'Yes'])
            TechSupport = st.selectbox("🛠️ Tech Support", ['No internet service', 'No', 'Yes'])
            StreamingTV = st.selectbox("📺 Streaming TV", ['No internet service', 'No', 'Yes'])
        
        with cols_services[2]:
            StreamingMovies = st.selectbox("🎬 Streaming Movies", ['No internet service', 'No', 'Yes'])

    st.write("")
    
    # Centered prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner('🔄 Analyzing customer data...'):
            # Map SeniorCitizen
            SeniorCitizen_mapped = 1 if SeniorCitizen_input == 'Yes' else 0
            
            # Get feature names
            num_features = preprocessor.transformers_[0][2]
            cat_features = preprocessor.transformers_[1][2]
            
            # Create input dictionary
            input_data = {
                'gender': gender,
                'SeniorCitizen': SeniorCitizen_mapped,
                'Partner': Partner,
                'Dependents': Dependents,
                'tenure': tenure,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity,
                'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection,
                'TechSupport': TechSupport,
                'StreamingTV': StreamingTV,
                'StreamingMovies': StreamingMovies,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges
            }
            
            all_features = num_features + cat_features
            
            try:
                input_df = pd.DataFrame([input_data])
                input_df = input_df[all_features]
                
                # Preprocess and predict
                input_processed = preprocessor.transform(input_df)
                prediction = model.predict(input_processed)[0]
                probability = model.predict_proba(input_processed)[0]
                
                # Display results
                st.markdown("---")
                
                if prediction == 1:
                    prob_churn = probability[1]
                    
                    # Risk assessment container
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{prob_churn*100:.1f}%</div>
                                <div class="stat-label">Churn Risk</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.error("### 🚨 HIGH CHURN RISK DETECTED")
                            st.markdown(f"""
                            This customer has a **{prob_churn*100:.1f}%** probability of churning. 
                            Immediate intervention recommended!
                            """)
                    
                    # Retention recommendations
                    with st.container(border=True):
                        st.subheader("💡 Recommended Retention Actions")
                        
                        recommendations = []
                        
                        if Contract == 'Month-to-month':
                            recommendations.append({
                                'icon': '📝',
                                'title': 'Contract Upgrade Opportunity',
                                'desc': 'Customer is on a month-to-month contract. Offer a **12-month contract with 10% discount** or **24-month contract with 20% discount** to increase retention.',
                                'priority': 'HIGH'
                            })
                        
                        if TechSupport == 'No' and InternetService != 'No':
                            recommendations.append({
                                'icon': '🛠️',
                                'title': 'Tech Support Bundle',
                                'desc': 'Customer lacks tech support. Bundle this service at a **15% discount** to improve satisfaction and reduce churn risk.',
                                'priority': 'HIGH'
                            })
                        
                        if InternetService == 'Fiber optic' and OnlineSecurity == 'No':
                            recommendations.append({
                                'icon': '🔒',
                                'title': 'Security Add-on',
                                'desc': 'Fiber optic customers without security are high-risk. Offer an **Online Security package** to address concerns.',
                                'priority': 'MEDIUM'
                            })
                        
                        if PaymentMethod == 'Electronic check':
                            recommendations.append({
                                'icon': '💳',
                                'title': 'Payment Method Optimization',
                                'desc': 'Electronic check payments correlate with higher churn. Incentivize switch to **automatic payment** with a small credit.',
                                'priority': 'MEDIUM'
                            })
                        
                        if tenure < 12:
                            recommendations.append({
                                'icon': '🎁',
                                'title': 'Early Customer Retention',
                                'desc': 'New customer (< 1 year). Provide **welcome gift or loyalty points** to build relationship and increase tenure.',
                                'priority': 'HIGH'
                            })
                        
                        if not recommendations:
                            recommendations.append({
                                'icon': '📞',
                                'title': 'Proactive Outreach',
                                'desc': 'Schedule a **customer satisfaction call** to understand concerns and offer personalized solutions.',
                                'priority': 'HIGH'
                            })
                        
                        for rec in recommendations:
                            priority_color = '#FF4B4B' if rec['priority'] == 'HIGH' else '#FFA500'
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, rgba(255,75,75,0.05), rgba(255,75,75,0.02)); 
                                        border-left: 4px solid {priority_color}; 
                                        padding: 1.5rem; 
                                        border-radius: 8px; 
                                        margin-bottom: 1rem;'>
                                <h4 style='margin: 0 0 0.5rem 0; color: #1a1a2e;'>
                                    {rec['icon']} {rec['title']} 
                                    <span style='background: {priority_color}; color: white; padding: 0.25rem 0.75rem; 
                                                 border-radius: 12px; font-size: 0.75rem; margin-left: 0.5rem;'>
                                        {rec['priority']} PRIORITY
                                    </span>
                                </h4>
                                <p style='margin: 0; color: #666;'>{rec['desc']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                else:
                    prob_stay = probability[0]
                    
                    # Low risk assessment
                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value" style="background: linear-gradient(135deg, #28a745, #20c997); 
                                                               -webkit-background-clip: text; 
                                                               -webkit-text-fill-color: transparent;">
                                    {prob_stay*100:.1f}%
                                </div>
                                <div class="stat-label">Retention Probability</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.success("### ✅ LOW CHURN RISK")
                            st.markdown(f"""
                            This customer has a **{prob_stay*100:.1f}%** probability of staying. 
                            Continue providing excellent service!
                            """)
                    
                    # Growth opportunities
                    with st.container(border=True):
                        st.subheader("📈 Growth Opportunities")
                        
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, rgba(40,167,69,0.05), rgba(32,201,151,0.02)); 
                                    border-left: 4px solid #28a745; 
                                    padding: 1.5rem; 
                                    border-radius: 8px;'>
                            <h4 style='margin: 0 0 0.5rem 0; color: #1a1a2e;'>🎯 Upsell Opportunities</h4>
                            <p style='margin: 0; color: #666;'>
                                This satisfied customer may be interested in:
                            </p>
                            <ul style='color: #666; margin-top: 0.5rem;'>
                                <li>Premium service tiers</li>
                                <li>Additional add-on services</li>
                                <li>Referral program enrollment</li>
                                <li>Loyalty rewards for long-term commitment</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ An error occurred during prediction: {e}")
                st.error("Please ensure all inputs are correct and try again.")

# --- ENHANCED FOOTER ---
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style='font-size: 1.1rem; margin-bottom: 1rem;'>
        <strong>Built with ❤️ by Rahul</strong>
    </p>
    <p>
        <a href="https://github.com/RahulDhaka29" target="_blank">🔗 GitHub</a> • 
        <a href="https://www.linkedin.com/in/rahul-dhaka-56b975289/" target="_blank">💼 LinkedIn</a> • 
        <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank">📊 Dataset</a>
    </p>
    <p style='margin-top: 1rem; font-size: 0.9rem; color: #999;'>
        Powered by Streamlit • Machine Learning • Python
    </p>
</div>
""", unsafe_allow_html=True)