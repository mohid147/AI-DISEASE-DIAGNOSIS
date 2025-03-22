import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Load dataset
df = pd.read_csv("dataset.csv")

# Convert symptoms into a list format
df["symptoms"] = df["symptoms"].apply(lambda x: x.split(","))

# One-hot encode symptoms
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])

y = df["disease"]  # Target variable (disease names)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & symptom encoder
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("symptom_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("Model training complete! Saved as 'disease_model.pkl'")
