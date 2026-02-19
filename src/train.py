from xml.parsers.expat import model
import joblib

from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train):

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    joblib.dump(model, "models/churn_model.pkl")

    print("Model saved")
