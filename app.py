import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🤖",
    layout="wide"
)

# -----------------------
# LOAD MODEL
# -----------------------

model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# -----------------------
# SIDEBAR
# -----------------------

st.sidebar.title("🤖 Customer Churn AI")

menu = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Batch Prediction", "Analytics"]
)

# -----------------------
# HOME HEADER
# -----------------------

st.title("Customer Churn Prediction System")

# ============================================================
# 1️⃣ SINGLE PREDICTION
# ============================================================

if menu == "Single Prediction":

    st.header("🔮 Single Customer Prediction")

    input_data = {}

    cols = st.columns(3)

    for i, col in enumerate(feature_columns):

        value = cols[i % 3].number_input(
            col,
            value=0.0
        )

        input_data[col] = value

    if st.button("Predict"):

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0][prediction]

        if prediction == 1:
            st.error(f"⚠️ Customer will CHURN")
        else:
            st.success(f"✅ Customer will STAY")

        st.write(f"Confidence: {probability:.2%}")

# ============================================================
# 2️⃣ BATCH PREDICTION
# ============================================================

elif menu == "Batch Prediction":

    st.header("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.write("Preview:")
        st.dataframe(df.head())

        prediction = model.predict(df)

        probability = model.predict_proba(df).max(axis=1)

        df["Prediction"] = prediction
        df["Confidence"] = probability

        st.success("Prediction Complete")

        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Result",
            csv,
            "prediction.csv",
            "text/csv"
        )

# ============================================================
# 3️⃣ ANALYTICS
# ============================================================

elif menu == "Analytics":

    st.header("📈 Analytics Dashboard")

    file = st.file_uploader(
        "Upload Prediction CSV",
        type=["csv"]
    )

    if file:

        df = pd.read_csv(file)

        st.subheader("Dataset Preview")

        st.dataframe(df)

        col1, col2 = st.columns(2)

        # Pie Chart

        with col1:

            fig = px.pie(
                df,
                names="Prediction",
                title="Churn Distribution"
            )

            st.plotly_chart(fig)

        # Confidence

        with col2:

            fig = px.histogram(
                df,
                x="Confidence",
                title="Confidence Distribution"
            )

            st.plotly_chart(fig)

        # Stats

        st.subheader("Statistics")

        st.write("Total Customers:", len(df))

        churn_rate = df["Prediction"].mean()

        st.write(f"Churn Rate: {churn_rate:.2%}")
