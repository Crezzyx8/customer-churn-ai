
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns


model, feature_columns = load_model()

st.sidebar.title("Customer Churn AI")

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Single Prediction",
        "📂 Batch Prediction",
        "📈 Analytics"
    ]
)

if menu == "🏠 Home":

    st.title("Customer Churn Prediction AI")

    st.markdown("### Professional AI Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.info("Machine Learning Model")
    col2.info("Batch Prediction")
    col3.info("Analytics Dashboard")

    st.divider()

    st.markdown("""
    ### Features

    ✅ Predict single customer  
    ✅ Upload CSV for batch prediction  
    ✅ Download prediction result  
    ✅ Interactive analytics  
    ✅ Confidence score  

    ---
    """)

elif menu == "Single Prediction":

    st.title("Single Customer Prediction")

    input_data = {}

    col1, col2 = st.columns(2)

    for i, col in enumerate(feature_columns):

        if i % 2 == 0:
            input_data[col] = col1.number_input(col, value=0.0)
        else:
            input_data[col] = col2.number_input(col, value=0.0)


    if st.button("Predict", use_container_width=True):

        df = pd.DataFrame([input_data])

        prediction = model.predict(df)[0]

        probability = model.predict_proba(df)[0]

        confidence = probability[prediction]


        st.divider()

        if prediction == 1:

            st.error("Customer will CHURN")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2%}"
            )

        else:

            st.success("Customer will STAY")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2%}"
            )

elif menu == "Batch Prediction":

    st.title("Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )


    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.subheader("Preview")

        st.dataframe(df, use_container_width=True)


        if st.button("Run Prediction", use_container_width=True):

            with st.spinner("Running prediction..."):


                predictions = model.predict(df)

                probabilities = model.predict_proba(df)

                confidence = probabilities.max(axis=1)


                df["Prediction"] = predictions

                df["Confidence"] = confidence


            st.success("Prediction Completed")


            st.subheader("Result")

            st.dataframe(df, use_container_width=True)


            csv = df.to_csv(index=False).encode("utf-8")


            st.download_button(

                label="Download CSV",

                data=csv,

                file_name="prediction_result.csv",

                mime="text/csv",

                use_container_width=True

            )

elif menu == "Analytics":

    st.title("Analytics Dashboard")

    uploaded_file = st.file_uploader(
        "Upload prediction result CSV",
        type=["csv"]
    )


    if uploaded_file:

        df = pd.read_csv(uploaded_file)


        if "Prediction" not in df.columns:

            st.error("Prediction column not found")

        else:

            total = len(df)

            churn = df["Prediction"].sum()

            churn_rate = churn / total


            col1, col2, col3 = st.columns(3)


            col1.metric(
                "Total Customers",
                total
            )

            col2.metric(
                "Churn Customers",
                churn
            )

            col3.metric(
                "Churn Rate",
                f"{churn_rate:.2%}"
            )


            st.divider()


            st.subheader("Churn Distribution")


            fig1 = px.pie(

                df,

                names="Prediction",

                title="Churn vs Stay"

            )

            st.plotly_chart(fig1, use_container_width=True)


            st.subheader("Confidence Distribution")


            fig2 = px.histogram(

                df,

                x="Confidence",

                nbins=20,

                title="Confidence Score"

            )

            st.plotly_chart(fig2, use_container_width=True)


st.divider()

st.caption("Customer Churn AI • Production Ready • Streamlit")
