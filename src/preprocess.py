import pandas as pd


def load_data(path):

    return pd.read_csv(path)


def preprocess_data(df):

    df = df.drop("customer_id", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    return df


def split_data(df):

    X = df.drop("churn", axis=1)

    y = df["churn"]

    return X, y
