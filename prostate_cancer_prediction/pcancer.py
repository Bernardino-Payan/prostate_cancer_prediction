mport numpy as np
import joblib
import os
from flask import Flask, request, jsonify, render_template

# Initialize Flask App
app = Flask(__name__)

# Load the trained model
MODEL_PATH = "final_prostate_cancer_rf_model.pkl"
model = joblib.load(MODEL_PATH)

# Define the top 10 features based on importance
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
                return jsonify({"error": f"Missing input: {feature_name}"})

        # Ensure it's formatted as a 2D array for model prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        # Convert prediction to readable format
        result = "High Risk" if prediction == 1 else "Low Risk"
        return render_template('index.html', prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)})

# API route for JSON-based requests
@app.route('/predict_json', methods=['POST'])
def predict_json():
    try:
        # Expecting JSON input
        data = request.get_json()

        # Extracting only the necessary top 10 features
        features = [float(data.get(feature, 0)) for feature in TOP_10_FEATURES]  # Default to 0 if missing

        # Ensure input is in the correct shape
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        # Convert prediction to readable format
        result = "High Risk" if prediction == 1 else "Low Risk"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
