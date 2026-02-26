import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

model = joblib.load("models/churn_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
}

.stButton>button {
    background-color:#00c6ff;
    color:white;
    border-radius:10px;
    height:3em;
    width:100%;
    font-size:18px;
}

.metric-card {
    background:#1b2a41;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

st.title("Customer Churn Prediction Using AI ")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (bulan)",0,100,12)

    with col2:
        monthly = st.number_input("Monthly Charges",0,10000,70)

    with col3:
        total = st.number_input("Total Charges",0,100000,1000)

    if st.button("Predict Now"):

        input_dict = {
            'tenure': tenure,
            'monthly_charges': monthly,
            'total_charges': total
        }

        input_df = pd.DataFrame([input_dict])

        for col in feature_columns:

            if col not in input_df.columns:

                input_df[col] = 0

        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        st.divider()

        col1,col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error("Customer will CHURN")

            else:

                st.success("Customer will STAY")

        with col2:

            st.metric("Churn Probability",f"{probability*100:.2f}%")

        fig = px.bar(
            x=["Stay","Churn"],
            y=[1-probability,probability],
            color=["Stay","Churn"]
        )

        st.plotly_chart(fig,use_container_width=True)



with tab2:

    file = st.file_uploader("Upload CSV",type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.write("Preview",df.head())


        for col in feature_columns:

            if col not in df.columns:

                df[col] = 0

        df = df[feature_columns]

        prediction = model.predict(df)

        probability = model.predict_proba(df)[:,1]

        result = df.copy()

        result["Prediction"] = prediction

        result["Probability"] = probability

        st.write("Result",result)

        csv = result.to_csv(index=False)

        st.download_button(

            "Download Result",

            csv,

            "prediction.csv",

            "text/csv"

        )

        fig = px.histogram(result,x="Probability")

        st.plotly_chart(fig,use_container_width=True)

