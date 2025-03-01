import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    args = parser.parse_args()

    print(f" Loading training data from {args.train}")

    # Verify if train directory exists
    if not os.path.exists(args.train):
        raise ValueError(f" Training directory not found: {args.train}")

    # Check if training file exists
    train_path = os.path.join(args.train, "prostate_cancer_prediction.csv")
    if not os.path.exists(train_path):
        raise ValueError(f" Training file not found: {train_path}")

    # Load data
    df = pd.read_csv(train_path)

    # Verify dataset
    print(f" Training data loaded: {df.shape} rows and {df.columns} columns")

    # Ensure the target column exists
    if "Biopsy_Result" not in df.columns:
        raise ValueError(" Missing target column 'Biopsy_Result' in dataset.")

    # Convert Biopsy_Result from categorical to numeric
    df["Biopsy_Result"] = df["Biopsy_Result"].map({"Benign": 0, "Malignant": 1})

    # Convert binary categorical variables to numeric (0 = No, 1 = Yes)
    binary_vars = [
        "Family_History", "Race_African_Ancestry", "Difficulty_Urinating", "Weak_Urine_Flow",
        "Blood_in_Urine", "Pelvic_Pain", "Back_Pain", "Erectile_Dysfunction", "Survival_5_Years",
        "Exercise_Regularly", "Healthy_Diet", "Smoking_History", "Hypertension", "Diabetes",
        "Follow_Up_Required", "Genetic_Risk_Factors", "Previous_Cancer_History", "Early_Detection"
    ]

    for var in binary_vars:
        if var in df.columns:
            df[var] = df[var].map({"No": 0, "Yes": 1})

    print(" Binary categorical variables converted to numeric.")

    # Apply one-hot encoding to categorical variables
    categorical_vars = ["Cancer_Stage", "Treatment_Recommended", "Alcohol_Consumption", "Cholesterol_Level"]

    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

    print(" One-hot encoding applied successfully!")

    # Define top 10 features (same as in your .ipynb)
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

    # Ensure all features exist
    missing_features = [feature for feature in TOP_10_FEATURES if feature not in df.columns]
    if missing_features:
        raise ValueError(f" Missing features in dataset: {missing_features}")

    # Define feature matrix
    X = df[TOP_10_FEATURES]

    # Extract target variable
    y = df["Biopsy_Result"]

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    # Save model
    model_path = os.path.join("/opt/ml/model", "model.pkl")
    joblib.dump(model, model_path)
    print(f" Model saved to {model_path}")
