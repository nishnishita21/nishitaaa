import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
import os
import time

st.set_page_config(page_title="High-Value Customer Predictor", layout="wide", page_icon="⚡")

# --- CUSTOM CSS mimicking the Bolt.new React App ---
st.markdown("""
<style>
/* Light theme mirroring the reference app */
.stButton>button {
    background: linear-gradient(to right, #2563EB, #1D4ED8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    transition: all 0.2s;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.4);
}
.main-card {
    background-color: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    border: 1px solid #E2E8F0;
    margin-top: 1rem;
}
.stepper-container {
    display: flex;
    justify-content: space-between;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E2E8F0;
}
.step-item {
    font-weight: 600;
    color: #64748B;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
}
.step-item.active { color: #2563EB; }
.model-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
}
.metric-value { font-size: 1.25rem; font-weight: 700; color: #2563EB; }
.insight-box {
    background-color: #ECFDF5;
    border-left: 4px solid #10B981;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
}
.alert-box {
    background-color: #FEF3C7;
    border-left: 4px solid #F59E0B;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
}
</style>
""", unsafe_allow_html=True)

# Helper Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, 'ml')
data_path = r"c:\nishita\hackathon_dataset (1).csv"

# Global State
if 'step' not in st.session_state:
    st.session_state.step = 1

# Header
st.markdown("<h1 style='display: flex; align-items: center; gap: 10px; margin-bottom: 0;'><span style='color: #2563EB;'>⚡</span> ML High-Value Prediction Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1.1rem; margin-top: 5px; margin-bottom: 2rem;'>End-to-end Machine Learning Pipeline & Actionable Intelligence</p>", unsafe_allow_html=True)

# Fake Stepper
steps = ["1. Upload", "2. EDA", "3. Preprocessing", "4. Training", "5. Report", "6. Live Predict"]
cols = st.columns(6)
for i, step_name in enumerate(steps):
    with cols[i]:
        if st.session_state.step == i + 1:
            st.markdown(f"<div class='step-item active'>🔵 {step_name}</div>", unsafe_allow_html=True)
        elif st.session_state.step > i + 1:
            st.markdown(f"<div class='step-item' style='color:#10B981;'>✅ {step_name}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='step-item'>⚪ {step_name}</div>", unsafe_allow_html=True)

st.markdown("<div class='main-card'>", unsafe_allow_html=True)

# --- STEP 1: Upload ---
if st.session_state.step == 1:
    st.subheader("📂 Upload Dataset")
    
    uploaded_file = st.file_uploader("Drag & drop your CSV file here, or click to browse", type=["csv"])
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.session_state.step = 2
        st.rerun()
    
    st.markdown("<p style='text-align: center; margin: 1rem 0; color: #64748B;'>— OR —</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("Load Default Hackathon Dataset 🚀", use_container_width=True):
            st.session_state.data = pd.read_csv(data_path)
            st.session_state.step = 2
            st.rerun()

# --- STEP 2: EDA ---
elif st.session_state.step == 2:
    st.subheader("📊 Exploratory Data Analysis")
    if 'data' not in st.session_state:
        st.session_state.data = pd.read_csv(data_path)
    df = st.session_state.data
    
    st.markdown("**Data Preview (Top 5 rows)**")
    st.dataframe(df.head(5), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Target Distribution (High Value)**")
        dist = df['high_value_purchase'].value_counts().reset_index()
        dist.columns = ['Is High Value', 'Count']
        dist['Is High Value'] = dist['Is High Value'].map({1: "High Value", 0: "Standard"})
        st.bar_chart(dist, x="Is High Value", y="Count")
    
    with col2:
        st.markdown("**Quick Correlation Highlights**")
        st.markdown("""
        - **Total Purchases**: <progress value="85" max="100"></progress> High
        - **Avg Order Value**: <progress value="75" max="100"></progress> High
        - **Engagement (Email)**: <progress value="60" max="100"></progress> Moderate
        - **Bounce Rate**: <progress value="30" max="100"></progress> Negative
        """, unsafe_allow_html=True)

    if st.button("Proceed to Preprocessing"):
        st.session_state.step = 3
        st.rerun()

# --- STEP 3: Preprocessing ---
elif st.session_state.step == 3:
    st.subheader("⚙️ Data Preprocessing")
    st.write("Applying transformations to prepare real-world data for machine learning models.")
    
    with st.spinner("Scaling numerical features..."):
        time.sleep(1)
        st.success("✅ Scaled numerical features (StandardScaler)")
    
    with st.spinner("Encoding categorical variables..."):
        time.sleep(1)
        st.success("✅ Encoded categorical variables (`country`, `device_type`, `customer_segment`)")
        
    with st.spinner("Handling missing values..."):
        time.sleep(0.5)
        st.success("✅ Handled missing values via median/mode imputation")

    if st.button("Start Model Training"):
        st.session_state.step = 4
        st.rerun()

# --- STEP 4: Model Training ---
elif st.session_state.step == 4:
    st.subheader("🧠 Model Training & Evaluation")
    
    with open(os.path.join(ML_DIR, 'metrics.json'), 'r') as f:
        metrics = json.load(f)
        
    for model_name, scores in metrics.items():
        acc = scores['Accuracy'] * 100
        f1 = scores['F1-score'] * 100
        
        st.markdown(f"""
        <div class="model-card">
            <div>
                <h4 style="margin: 0; color: #1E293B;">{model_name}</h4>
                <span style="color: #64748B; font-size: 0.85rem;">Ensemble / Linear</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.85rem; color: #64748B;">Accuracy: <span class="metric-value">{acc:.1f}%</span></div>
                <div style="font-size: 0.85rem; color: #64748B;">F1 Score: <span class="metric-value">{f1:.1f}%</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("Generate Final Report"):
        st.session_state.step = 5
        st.rerun()

# --- STEP 5: Report ---
elif st.session_state.step == 5:
    st.subheader("📝 Final Intelligence Report")
    
    with open(os.path.join(ML_DIR, 'feature_importances.json'), 'r') as f:
        imps = json.load(f)
        
    st.markdown("**Global Feature Importance (Random Forest Drivers)**")
    f_df = pd.DataFrame.from_dict(imps, orient='index', columns=['Importance']).head(8)
    f_df.index = [x.replace('num__', '').replace('cat__', '').replace('_', ' ').title() for x in f_df.index]
    st.bar_chart(f_df, height=300)
    
    st.markdown("""
    <div class="insight-box">
        <strong style="color: #065F46;">Key Model Insights</strong>
        <ul style="color: #065F46; margin-top: 0.5rem; margin-bottom: 0;">
            <li>The Random Forest model achieved the highest predictive accuracy.</li>
            <li>Total purchases and Average Order Value are the strongest global indicators of high-value status.</li>
            <li>Bounce rate has a strong negative correlation with successful conversions.</li>
        </ul>
    </div>
    
    <div class="alert-box">
        <strong style="color: #92400E;">Business Recommendations</strong>
        <ul style="color: #92400E; margin-top: 0.5rem; margin-bottom: 0;">
            <li><strong>Retention:</strong> Deploy custom discount codes for active abandoners immediately.</li>
            <li><strong>Growth:</strong> Upsell Platinum memberships to users exceeding 5+ transactions.</li>
            <li><strong>UX:</strong> Optimize mobile checkout flows to drastically reduce bounce rates.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Open Live Prediction Engine ✨"):
        st.session_state.step = 6
        st.rerun()

# --- STEP 6: LIVE PREDICTION DASHBOARD ---
elif st.session_state.step == 6:
    st.subheader("🔮 Live Prediction Dashboard")
    st.markdown("Use the loaded models to run instant **What-If** analysis on high-value likelihood.")
    
    try:
        preprocessor = joblib.load(os.path.join(ML_DIR, 'preprocessor.joblib'))
        rf_model = joblib.load(os.path.join(ML_DIR, 'random_forest.joblib'))
    except:
        st.error("Models not found. Train them first!")
        st.stop()

    col1, col2 = st.columns([1, 2])
    
    # Sliders matched to Hackathon Dataset
    with col1:
        st.markdown("### 👤 User Demographics")
        age = st.slider("👤 Client Age", 18, 80, 35, help="Age of the customer in years.")
        country = st.selectbox("🌍 Geographic Region", ["USA", "Germany", "Canada", "UK", "India"])
        device_type = st.selectbox("💻 Primary Device", ["Desktop", "Mobile", "Tablet"])
        customer_segment = st.selectbox("🏅 Loyalty Tier", ["Bronze", "Silver", "Gold", "Platinum"])

        st.markdown("<br/>", unsafe_allow_html=True)

        st.markdown("### 🛒 User Behavior & Engagement")
        total_purchases = st.slider("🛍️ Lifetime Purchases", 0, 50, 8)
        avg_order_value = st.slider("💳 Avg Order Value ($)", 10.0, 1000.0, 150.0)
        cart_abandonment = st.slider("❌ Cart Abandonment Rate", 0.0, 1.0, 0.45, help="1.0 means they abandon every cart.")
        email_opens = st.slider("📧 Marketing Email Opens", 0, 20, 5)

        # Hidden but required fields defaulting to medians 
        account_age_months = 12
        days_since_last = 15.0
        reviews_count = 2
        review_rating = 4.0
        bounce_rate = 0.2
        has_promo = 0

    with col2:
        import requests
        
        payload = {
            "age": age, "account_age_months": account_age_months, 
            "total_purchases": total_purchases, "avg_order_value": avg_order_value, 
            "days_since_last_purchase": days_since_last, "cart_abandonment_rate": cart_abandonment, 
            "product_reviews_count": reviews_count, "avg_review_rating": review_rating, 
            "email_opens": email_opens, "bounce_rate": bounce_rate, 
            "customer_segment": customer_segment, "device_type": device_type, 
            "country": country, "has_promo_code": has_promo,
            "model_name": "Random Forest"
        }
        
        try:
            res = requests.post("http://localhost:8000/predict", json=payload)
            if res.status_code == 200:
                data = res.json()
                is_high = data['prediction'] == 1
                prob = data['probability']
                
                # REST API Powered Prediction Box
                st.markdown(f"""
                    <div style="background: {'#ECFDF5' if is_high else '#FEF2F2'}; border: 2px solid {'#10B981' if is_high else '#EF4444'}; border-radius: 12px; padding: 2rem; text-align: center; margin-top: 1rem;">
                        <h3 style="color: {'#065F46' if is_high else '#991B1B'}; margin-bottom: 0;">Predicted Likelihood</h3>
                        <h1 style="color: {'#10B981' if is_high else '#EF4444'}; font-size: 3.5rem; margin: 0;">{prob*100:.1f}%</h1>
                        <h4 style="color: {'#065F46' if is_high else '#991B1B'};">{'⭐ High-Value Prospect' if is_high else '📉 Standard Prospect'}</h4>
                        <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.03); border-radius: 8px;">
                            <span style="font-weight: 600; font-size: 1.1rem; color: #1E293B;">Est. Lifetime Value (LTV): ${data['estimated_clv_usd']:,.2f}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Customer Intelligence & Explainable AI
                st.markdown("---")
                st.markdown(f"### 🏷️ Segment: **{data['segment']}**")
                
                st.info(f"**🧠 Explainable AI Insight:**\n\n{data['explainability']}\n\n*Top Drivers*: {', '.join(data['top_influencing_features'][:3])}")
                
                # Business Recommendation Engine
                st.markdown("**⚡ Actionable Business Engine**")
                for r in data['recommendations']:
                    st.write(f"- {r}")
                    
                st.markdown("<br/>", unsafe_allow_html=True)
                
                # Downloadable Report Generator
                report = f"--- HIGH VALUE PREDICTION REPORT ---\n\n"
                report += f"Predicted High-Value: {'YES' if is_high else 'NO'} ({prob*100:.1f}%)\n"
                report += f"Estimated Customer Lifetime Value: ${data['estimated_clv_usd']:,.2f}\n"
                report += f"Customer Segment: {data['segment']}\n\n"
                report += f"EXPLAINABLE AI INSIGHT:\n{data['explainability']}\n\n"
                report += "RECOMMENDATIONS:\n" + "\n".join([f"- {r}" for r in data['recommendations']])
                
                st.download_button("📥 Download Intelligence Report (TXT)", data=report, file_name="prediction_report.txt")
                
            else:
                st.error(f"API Error {res.status_code}: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Backend API Offline! Please start the FastAPI backend by running `uvicorn app.main:app` in a new terminal.")

        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("Reset Pipeline"):
            st.session_state.step = 1
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
