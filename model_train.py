
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv(r"D:\hckk\synthetic_consumers_1_50 (2).csv")  

print("Initial dataset shape:", df.shape)
print(df.head())

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print("Shape after cleaning:", df.shape)

features = ["daily_kwh", "cumulative_kwh", "plan_kwh", "remaining_credit", "usage_ratio"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")  
model.fit(X_scaled)
print("One-Class SVM trained successfully")

preds = model.predict(X_scaled)
print("Normal samples :", (preds == 1).sum())
print("Outlier samples:", (preds == -1).sum())

joblib.dump(model, "hes_ocsvm_model.pkl")
joblib.dump(scaler, "hes_scaler.pkl")
print("Model and scaler saved successfully")

new_sample = pd.DataFrame([{
    "consumer_id": "C101",         
    "daily_kwh": 20,
    "cumulative_kwh": 250,
    "plan_kwh": 500,
    "remaining_credit": 250,
    "usage_ratio": 20/500
}])

X_new = scaler.transform(new_sample[features])
prediction = model.predict(X_new)[0]
score = model.decision_function(X_new)[0]
status = "Anomaly" if prediction == -1 else "Normal"

print(f"Consumer ID: {new_sample['consumer_id'].iloc[0]}")
print(f"Prediction: {status}, Score: {score:.3f}")