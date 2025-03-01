import argparse
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def train_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Define features and target variable
    X = df.drop(columns=['Biopsy_Result'])  # Assuming 'Biopsy_Result' is the target
    y = df['Biopsy_Result'].map({"Benign": 0, "Malignant": 1})  # Convert to binary

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = os.path.join("/opt/ml/model", "model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")  # SageMaker training input path
    args = parser.parse_args()

    train_model(args.train)
