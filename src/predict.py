import joblib

from preprocess import load_data, preprocess_data


model = joblib.load("models/churn_model.pkl")


df = load_data("../data/churn.csv")

df = preprocess_data(df)


sample = df.drop("churn", axis=1).iloc[[0]]


result = model.predict(sample)


print(result)
