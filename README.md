# Prostate Cancer Prediction

This project focuses on predicting the likelihood of **prostate cancer** using machine learning models. The dataset used for training was obtained from **[Kaggle](https://www.kaggle.com/datasets/ankushpanday1/prostate-cancer-prediction-dataset)** and contains **27,945 records with 30 features**. The model has been deployed in multiple environments, including **Flask (local deployment), Docker (containerized application), Heroku (CI/CD pipeline deployment), AWS SageMaker (cloud-based training and deployment), and Kubernetes (orchestration for containerized deployment)**.

## Project Workflow

1. **Define the Problem** - Predict **prostate cancer risk** based on patient features. Machine learning is ideal as it can analyze patterns and relationships between medical features.  
2. **Data Exploration & Preprocessing** - Data cleaning and feature engineering were performed in **Jupyter Notebook (AWS SageMaker Studio Lab)**. Applied **one-hot encoding**, handled missing values, and transformed categorical features.  
3. **Model Training & Evaluation** - Used **Random Forest** and **XGBoost** classifiers. Addressed class imbalance using **SMOTE**. Evaluated models using **Precision, Recall, F1-score, ROC-AUC, Confusion Matrix**.  
4. **Deployment Steps** - **Flask** for local deployment, **Docker** to containerize the application, **Heroku** for cloud deployment using a **CI/CD pipeline**, **AWS SageMaker** for model training and cloud-based hosting, and **Kubernetes** for scaling and managing containers.  

## Dataset Overview

The dataset contains the following key features:

- `Age`: Patient's age  
- `PSA_Level`: Prostate-Specific Antigen level  
- `DRE_Result`: Digital Rectal Exam result (Normal/Abnormal)  
- `Biopsy_Result`: Target Variable (Benign/Malignant)  
- `Cancer_Stage`: Localized, Metastatic, or Advanced  
- `Treatment_Recommended`: Various treatment options like Surgery, Radiation, etc.  
- `Smoking_History`: Smoking habits of the patient  
- `Diabetes`: Whether the patient has diabetes (Yes/No)  
- `Alcohol_Consumption`: Levels of alcohol consumption (Low/Moderate/High)  

## Project Files

- `AWS_pcancer.ipynb` - Jupyter Notebook for AWS SageMaker training and deployment  
- `train.py` - Training script used in AWS SageMaker  
- `pcancer.py` - Flask API for serving predictions  
- `Dockerfile` - Configuration file to containerize the application  
- `requirements.txt` - Dependencies needed for the project  
- `deployment.yaml` - Kubernetes deployment configuration  
- `service.yaml` - Kubernetes service configuration  
- `final_prostate_cancer_rf_model.pkl` - Trained machine learning model  

## Deployment Instructions

To run the project locally, using Docker, deploying it on Heroku, AWS SageMaker, and Kubernetes, follow these steps:

### Flask Local Deployment
```bash
git clone https://github.com/Bernardino-Payan/prostate_cancer_prediction.git
cd prostate_cancer_prediction
pip install -r requirements.txt
python pcancer.py
```
http://127.0.0.1:5001  

### Docker Deployment
```bash
docker build -t bernardinopayan/prostate-cancer-prediction .
docker run -p 5001:5001 bernardinopayan/prostate-cancer-prediction
```

### CI/CD Deployment on Heroku
```bash
heroku login
heroku container:login
docker tag bernardinopayan/prostate-cancer-prediction registry.heroku.com/prostate-cancer-api/web
docker push registry.heroku.com/prostate-cancer-api/web
heroku container:release web -a prostate-cancer-api
heroku open -a prostate-cancer-api
```

### AWS SageMaker Deployment
```python
import boto3
s3 = boto3.client("s3")
bucket_name = "my-prostate-cancer-bucket"
file_key = "prostate_cancer_prediction.csv"
s3.upload_file("prostate_cancer_prediction.csv", bucket_name, file_key)

from sagemaker.sklearn.estimator import SKLearn
sklearn = SKLearn(
    entry_point="train.py",
    instance_type="ml.m4.xlarge",
    framework_version="1.2-1",
    py_version="py3",
    role=role,
    sagemaker_session=sagemaker_session
)
sklearn.fit({"train": f"s3://{bucket_name}/prostate_cancer_prediction.csv"})

deployment = sklearn.deploy(instance_type="ml.m4.xlarge", initial_instance_count=1)
```

### Kubernetes Deployment
```yaml
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
          image: bernardinopayan/prostate_cancer_api:latest
          ports:
            - containerPort: 5001
```

```yaml
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
```

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

## Model Performance

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Accuracy | 51% | 51% |
| ROC-AUC Score | 0.513 | 0.511 |
| Precision | 51% | 51% |
| Recall | 54% | 60% |

## Author

Bernardino Payan  
Email: b.payan8432@student.nu.edu  
National University | Student ID: 9000684321  

## References

- Dataset: [Prostate Cancer Prediction - Kaggle](https://www.kaggle.com/datasets/ankushpanday1/prostate-cancer-prediction-dataset)  
- SageMaker Docs: [AWS SageMaker Training](https://docs.aws.amazon.com/sagemaker/latest/dg/train-model.html)  

## Conclusion

This project successfully demonstrates end-to-end machine learning model deployment, integrating Flask, Docker, Heroku, AWS SageMaker, and Kubernetes.
