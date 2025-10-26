import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

df = pd.read_csv("data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Accuracy:", acc)

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Local MLflow UI
mlflow.set_experiment("heart-mlops-demo")

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")
    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="HeartDiseaseModel"
    )

joblib.dump(model, "models/model.pkl")
print("Saved latest model → models/model.pkl")
