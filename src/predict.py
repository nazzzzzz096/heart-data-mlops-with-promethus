# src/predict.py
import joblib
import sys
import pandas as pd

# Load model
model = joblib.load("models/model.pkl")

# Example input from command line
# python src/predict.py 63 145 233 1 150 0 2 etc...
inputs = list(map(float, sys.argv[1:]))

# Convert to dataframe with correct number of columns
df = pd.DataFrame([inputs])

prediction = model.predict(df)[0]
print("❤️ Heart Disease Prediction:", "Has Disease" if prediction == 1 else "No Disease")
