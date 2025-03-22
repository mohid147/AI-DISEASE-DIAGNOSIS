from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset.csv")

# Extract diseases and symptoms
diseases = df["disease"].dropna().unique().tolist()
symptoms = set()
for symptom_list in df["symptoms"].dropna():
    symptoms.update(symptom_list.split(","))
symptoms = sorted(symptoms)

@app.route('/')
def home():
    return render_template('index.html', diseases=diseases, symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_disease = request.form.get("disease")
    selected_symptoms = request.form.get("symptoms")
    
    # Find matching symptoms for the selected disease
    matching_row = df[df["disease"] == selected_disease]
    if not matching_row.empty:
        predicted_symptoms = matching_row.iloc[0]["symptoms"]
    else:
        predicted_symptoms = "No data available"
    
    return render_template('index.html', diseases=diseases, symptoms=symptoms, result=predicted_symptoms)

if __name__ == '__main__':
    app.run(debug=True)
