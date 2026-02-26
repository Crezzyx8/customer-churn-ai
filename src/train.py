import joblib
import os

def train_model(X_train, y_train):

    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/churn_model.pkl")

    joblib.dump(X_train.columns.tolist(), "models/feature_columns.pkl")

    print("Model saved")
