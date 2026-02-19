import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🤖",
    layout="centered"
)

# Load model
model = joblib.load("models/churn_model.pkl")

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# Title
st.markdown('<p class="title">🤖 Customer Churn Prediction</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">AI Model to predict customer churn risk</p>', unsafe_allow_html=True)

st.divider()


# Input section
st.subheader("📊 Customer Information")

col1, col2 = st.columns(2)

with col1:

    tenure = st.slider("Tenure (Months)", 0, 72, 12)

    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)


with col2:

    total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)


st.divider()


# Prediction
if st.button("🔍 Predict", use_container_width=True):

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    prediction = model.predict(input_data)[0]


    if prediction == 1:

        st.markdown(
            '<div class="result-box" style="background-color:#ff4b4b;">⚠️ High Risk: Customer will Churn</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="result-box" style="background-color:#00c853;">✅ Safe: Customer will Stay</div>',
            unsafe_allow_html=True
        )


st.divider()

st.caption("Built with Python, Scikit-learn, and Streamlit")
