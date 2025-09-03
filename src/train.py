# src/train.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from features import extract_features

# Dataset folder structure
folders = {"healthy": 0, "rust": 1, "spot": 2}

X, y = [], []

# Loop over dataset
for label, value in folders.items():
    folder_path = f"data/{label}"
    for file in os.listdir(folder_path):
        fpath = os.path.join(folder_path, file)
        features = extract_features(fpath)
        X.append(features)
        y.append(value)

X = np.array(X)
y = np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=folders.keys()))

# Save model
joblib.dump(model, "mulberry_model.pkl")
print("✅ Model saved as mulberry_model.pkl")
