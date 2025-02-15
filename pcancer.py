import numpy as np
import joblib
import os
from flask import Flask, request, jsonify, render_template

# Initialize Flask App
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "final_prostate_cancer_rf_model.pkl"
model = joblib.load(MODEL_PATH)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form submission
        features = [float(request.form[f'feature_{i}']) for i in range(10)]  # Adjust for the top 10 features
        prediction = model.predict([features])[0]
        
        # Convert prediction to readable format
        result = "High Risk" if prediction == 1 else "Low Risk"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

