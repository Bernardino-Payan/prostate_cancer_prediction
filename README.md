# Prostate Cancer Prediction API

This repository contains a machine learning-based API for predicting prostate cancer risk. The API is built using Flask, containerized with Docker, and deployed using Kubernetes on Minikube. Additionally, the model is deployed on Heroku for cloud-based accessibility.

## Project Overview

### Problem Statement
Prostate cancer is a major health concern, and early detection is critical for improving treatment outcomes. This project aims to provide a predictive model based on patient health indicators to assess the risk of prostate cancer.

### Machine Learning Model
A Random Forest Classifier was trained using scikit-learn, with preprocessing steps such as feature selection, hyperparameter tuning, and handling class imbalance using SMOTE. The trained model is exposed via a Flask API.

## Repository Structure

prostate_cancer_prediction/
│── deployment.yaml                 # Kubernetes Deployment Configuration
│── service.yaml                     # Kubernetes Service Configuration
│── Dockerfile                        # Docker container configuration
│── requirements.txt                   # Python dependencies
│── Procfile                           # Heroku deployment configuration
│── pcancer.py                         # Flask API implementation
│── Payan_Bernardino_Prostate_Cancer.ipynb # Jupyter Notebook (Model Training)
│── final_prostate_cancer_rf_model.pkl  # Trained Machine Learning Model
│── minikube-linux-amd64                # Minikube binary (if necessary)
│── README.md                           # Project Documentation

## Setup and Deployment

### 1. Clone the Repository
git clone https://github.com/Bernardino-Payan/prostate_cancer_prediction.git
cd prostate_cancer_prediction

### 2. Install Dependencies and Run the API Locally
pip install -r requirements.txt
python pcancer.py

The API will be accessible at:
http://127.0.0.1:5001

### 3. Build and Run the Docker Container
docker build -t bernardinopayan/prostate-cancer-prediction:latest .
docker run -p 5001:5001 bernardinopayan/prostate-cancer-prediction

The API will be available at:
http://127.0.0.1:5001

## Deploying on Kubernetes (Minikube)

### 1. Install Minikube (If Necessary)
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube

### 2. Start Minikube
minikube start --driver=docker

### 3. Apply Kubernetes Configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

### 4. Check Deployment and Services
kubectl get pods
kubectl get services

### 5. Get Minikube’s IP Address
minikube ip

Then access the API at:
http://<minikube-ip>:30007

## Deploying on Heroku

### 1. Log in to Heroku
heroku login

### 2. Log in to Heroku Container Registry
heroku container:login

### 3. Set the Correct Heroku Remote
heroku git:remote -a prostate-cancer-api

### 4. Build and Push the Docker Image to Heroku
docker build --provenance=false -t prostate-cancer-prediction .
docker tag prostate-cancer-prediction registry.heroku.com/prostate-cancer-api/web
docker push registry.heroku.com/prostate-cancer-api/web

### 5. Release the Image on Heroku
heroku container:release web

### 6. Open and Test the API
heroku open

Check if the API is running properly:
curl -X GET https://prostate-cancer-api-6eefc0ced3b3.herokuapp.com/

For a prediction request, use:
curl -X POST https://prostate-cancer-api-6eefc0ced3b3.herokuapp.com/predict_json -H "Content-Type: application/json" -d '{
  "Alcohol_Consumption_Moderate": 1,
  "Alcohol_Consumption_Low": 0,
  "Treatment_Recommended_Chemotherapy": 1,
  "Treatment_Recommended_Hormone Therapy": 0,
  "Cancer_Stage_Localized": 1,
  "Treatment_Recommended_Surgery": 0,
  "Treatment_Recommended_Immunotherapy": 0,
  "Treatment_Recommended_Radiation": 1,
  "Diabetes": 0,
  "Cancer_Stage_Metastatic": 0
}'

## Kubernetes Configuration Files

### Deployment Configuration (deployment.yaml)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prostate-cancer-prediction
spec:
  replicas: 2
  selector:
    matchLabels:
      app: prostate-cancer
  template:
    metadata:
      labels:
        app: prostate-cancer
    spec:
      containers:
      - name: prostate-cancer
        image: bernardinopayan/prostate-cancer-prediction:latest
        ports:
        - containerPort: 5001

### Service Configuration (service.yaml)
apiVersion: v1
kind: Service
metadata:
  name: prostate-cancer-service
spec:
  selector:
    app: prostate-cancer
  ports:
    - protocol: TCP
      port: 5001
      targetPort: 5001
      nodePort: 30007
  type: NodePort

## Next Steps

- Set up a CI/CD pipeline to automate deployment to Heroku.
- Deploy the application to a cloud-based Kubernetes service such as AWS EKS, GCP GKE, or Azure AKS.
- Enable autoscaling for better resource management.

## Author

Bernardino Payan  
GitHub: [Bernardino-Payan](https://github.com/Bernardino-Payan)
