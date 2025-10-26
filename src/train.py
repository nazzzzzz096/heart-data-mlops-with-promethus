import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow

# Use local MLflow file store
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("heart-mlops-demo")

# Load dataset
df = pd.read_csv("data/heart.csv")
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict & calculate accuracy
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")

    # Log metric & model to MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 100)
    joblib.dump(model, "models/model.pkl")
    mlflow.log_artifact("models/model.pkl")
