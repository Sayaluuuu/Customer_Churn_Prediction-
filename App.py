import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model and scaler
scaler = joblib.load("sclaer.pkl")
model  = joblib.load("Customer_churn_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3d4470;
        text-align: center;
    }
    .churn-high   { border-left: 5px solid #ff4b4b; }
    .churn-medium { border-left: 5px solid #ffa500; }
    .churn-low    { border-left: 5px solid #00cc88; }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #a0aec0;
        margin-bottom: 10px;
        border-bottom: 1px solid #3d4470;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=80)
    st.title("Customer Churn App")
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.info("**Model:** Random Forest Classifier")

    st.markdown("### 📈 Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy",  "99.95%")
    col2.metric("ROC-AUC",   "0.997")

    st.markdown("### 📋 Features Used")
    features = ["Age", "Tenure", "Monthly Charges",
                "Total Charges", "Contract Type",
                "Internet Service", "Tech Support"]
    for f in features:
        st.markdown(f"- {f}")

# ── Header ───────────────────────────────────────────────
st.title(" Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn using a trained **Random Forest** model.")
st.markdown("---")

# ── Input Form ───────────────────────────────────────────
st.markdown('<div class="section-header">🧾 Customer Details</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
with col2:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=150, value=24)
with col3:
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=75.0)
with col4:
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=1800.0)

col5, col6, col7, col8 = st.columns(4)
with col5:
    gender = st.selectbox("Gender", ["Female", "Male"])
with col6:
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One-Year", "Two-Year"])
with col7:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "None"])
with col8:
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])

st.markdown("---")

# ── Predict Button ────────────────────────────────────────
predict_btn = st.button("🔍 Predict Churn", use_container_width=True)

if predict_btn:

    input_data = pd.DataFrame({
        "Age":                          [age],
        "Tenure":                       [tenure],
        "MonthlyCharges":               [monthly_charges],
        "TotalCharges":                 [total_charges],
        "Gender_Male":                  [1 if gender == "Male" else 0],
        "ContractType_One-Year":        [1 if contract == "One-Year" else 0],
        "ContractType_Two-Year":        [1 if contract == "Two-Year" else 0],
        "InternetService_Fiber Optic":  [1 if internet_service == "Fiber Optic" else 0],
        "InternetService_None":         [1 if internet_service == "None" else 0],
        "TechSupport_Yes":              [1 if tech_support == "Yes" else 0]
    })

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]
    retention    = 1 - probability

    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Prediction Results</div>', unsafe_allow_html=True)

    # ── Result Banner ─────────────────────────────────────
    if prediction == 1:
        st.error(f"⚠️ This customer is **likely to churn** — Churn Probability: **{probability:.2%}**")
    else:
        st.success(f" This customer is **likely to stay** — Retention Probability: **{retention:.2%}**")

    # ── Metrics Row ───────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability",     f"{probability:.2%}")
    m2.metric("Retention Probability", f"{retention:.2%}")
    m3.metric("Risk Level",
              "🔴 High" if probability > 0.8 else "🟡 Moderate" if probability > 0.5 else "🟢 Low")

    st.markdown("---")

    # ── Charts Row ────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Churn vs Retention")
        fig_pie = go.Figure(go.Pie(
            labels=["Churn", "Retention"],
            values=[probability, retention],
            hole=0.55,
            marker=dict(colors=["#ff4b4b", "#00cc88"]),
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(t=10, b=10),
            annotations=[dict(text=f"{probability:.0%}", x=0.5, y=0.5,
                              font_size=22, showarrow=False, font_color="white")]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Risk Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(probability * 100, 1),
            number=dict(suffix="%", font=dict(color="white")),
            delta=dict(reference=50, increasing=dict(color="#ff4b4b"),
                                     decreasing=dict(color="#00cc88")),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="white",
                          tickfont=dict(color="white")),
                bar=dict(color="#ff4b4b" if probability > 0.5 else "#00cc88"),
                bgcolor="rgba(0,0,0,0)",
                steps=[
                    dict(range=[0,  50], color="#1a2e1a"),
                    dict(range=[50, 80], color="#2e2a1a"),
                    dict(range=[80,100], color="#2e1a1a"),
                ],
                threshold=dict(line=dict(color="white", width=2), value=50)
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(t=20, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Feature Summary Table ─────────────────────────────
    st.markdown("#### 📋 Input Summary")
    summary = pd.DataFrame({
        "Feature": ["Age", "Tenure", "Monthly Charges", "Total Charges",
                    "Gender", "Contract", "Internet Service", "Tech Support"],
        "Value":   [age, tenure, f"${monthly_charges:.2f}", f"${total_charges:.2f}",
                    gender, contract, internet_service, tech_support]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)