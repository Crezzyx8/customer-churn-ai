import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from database import create_table, insert_prediction, get_history, delete_history

st.set_page_config(
    page_title="Customer Churn System",
    layout="wide"
)

create_table()

model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


st.sidebar.title("Customer Churn System")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Batch Prediction",
        "Analytics",
        "History"
    ]
)

st.title("Customer Churn Prediction Platform")

if menu == "Prediction":

    st.subheader("Single Customer Prediction")

    input_data = {}

    cols = st.columns(3)

    for i, col in enumerate(feature_columns):
        value = cols[i % 3].number_input(col, value=0.0)
        input_data[col] = value

    if st.button("Run Prediction"):

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][prediction]

        insert_prediction(prediction, float(probability))

        if prediction == 1:
            st.error("Customer is likely to churn")
        else:
            st.success("Customer is likely to stay")

        st.write(f"Confidence: {probability:.2%}")


elif menu == "Batch Prediction":

    st.subheader("Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.dataframe(df.head())

        if st.button("Process"):

            predictions = model.predict(df)
            probabilities = model.predict_proba(df).max(axis=1)

            df["Prediction"] = predictions
            df["Confidence"] = probabilities

            # Save to DB
            for p, c in zip(predictions, probabilities):
                insert_prediction(int(p), float(c))

            st.success("Prediction completed")

            st.dataframe(df)

            csv = df.to_csv(index=False)

            st.download_button(
                "Download Result",
                csv,
                "prediction_result.csv",
                "text/csv"
            )


elif menu == "Analytics":

    st.subheader("Analytics Dashboard")

    df = get_history()

    if df.empty:
        st.warning("No prediction history available")
    else:

        col1, col2, col3 = st.columns(3)

        total = len(df)
        churn = df["prediction"].sum()
        churn_rate = churn / total if total > 0 else 0

        col1.metric("Total Predictions", total)
        col2.metric("Churn Predictions", churn)
        col3.metric("Churn Rate", f"{churn_rate:.2%}")

        st.divider()

        fig1 = px.pie(
            df,
            names="prediction",
            title="Prediction Distribution"
        )

        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(
            df,
            x="confidence",
            nbins=20,
            title="Confidence Distribution"
        )

        st.plotly_chart(fig2, use_container_width=True)



elif menu == "History":

    st.subheader("Prediction History")

    df = get_history()

    if df.empty:
        st.info("No history found")
    else:
        st.dataframe(df)

        if st.button("Clear History"):
            delete_history()
            st.success("History cleared")
