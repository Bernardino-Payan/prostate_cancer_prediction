import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# Initialize Flask App
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "final_prostate_cancer_rf_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the top 10 features
TOP_10_FEATURES = [
    "Alcohol_Consumption_Moderate",
    "Alcohol_Consumption_Low",
    "Treatment_Recommended_Chemotherapy",
    "Treatment_Recommended_Hormone Therapy",
    "Cancer_Stage_Localized",
    "Treatment_Recommended_Surgery",
    "Treatment_Recommended_Immunotherapy",
    "Treatment_Recommended_Radiation",
    "Diabetes",
    "Cancer_Stage_Metastatic"
]

# Print Expected Features on API Start
print("Prostate Cancer Prediction API is running!")
print("Expected input features for predictions:")
print(TOP_10_FEATURES)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route (HTML Form Submission)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form inputs for the top 10 features
        features = []
        for feature_name in TOP_10_FEATURES:
            if feature_name in request.form:
                features.append(float(request.form[feature_name]))
            else:
                return jsonify({"error": f"Missing input: {feature_name}"}), 400

        # Ensure it's formatted as a 2D array for model prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        # Convert prediction to readable format
        result = "High Risk" if prediction == 1 else "Low Risk"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API route for JSON-based requests
@app.route('/predict_json', methods=['POST'])
def predict_json():
    try:
        # Expecting JSON input
        data = request.get_json()

        # Ensure all required features are present
        missing_features = [feature for feature in TOP_10_FEATURES if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing input features: {missing_features}"}), 400

        # Convert input data into a pandas DataFrame (fix for column issue)
        input_df = pd.DataFrame([data], columns=TOP_10_FEATURES)

        # Make a prediction
        prediction = model.predict(input_df)[0]

        # Convert prediction to readable format
        result = "High Risk" if prediction == 1 else "Low Risk"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))  # Use Heroku's dynamic port
    print(f" Starting Flask Server on http://0.0.0.0:{port}/")
    app.run(debug=True, host="0.0.0.0", port=port)

