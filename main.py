import sys
import os

# FIX PATH
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.preprocess import load_data, preprocess_data, split_data
from src.train import train_model
from src.evaluate import evaluate_model

from sklearn.model_selection import train_test_split


print("Loading data...")
df = load_data("data/churn.csv")

print("Preprocessing...")
df = preprocess_data(df)

X, y = split_data(df)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training model...")
train_model(X_train, y_train)


print("Evaluating model...")
evaluate_model(X_test, y_test)
