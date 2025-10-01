## Loan Approval Prediction 🏦✅
An end‑to‑end machine learning project that predicts whether a loan application should be approved based on applicant details. The model is trained on the LoanApproval.csv dataset using 9 key features:
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
The project integrates data science, machine learning, and full‑stack deployment with:
- Random Forest Classifier with GridSearchCV for hyperparameter tuning
- FastAPI backend with Uvicorn server
- Frontend UI for user interaction
- Docker for containerization
- CI/CD pipeline for automated testing and deployment
- pytest for unit and integration testing

## 📌 Project Objective
The goal is to build a loan approval prediction system that:
- Analyzes applicant data to predict loan approval status
- Provides an intuitive web interface for users and loan officers
- Ensures reproducibility and scalability with Docker and CI/CD

## 🧪 Methodology
- Data Understanding – Explore the LoanApproval.csv dataset
- EDA – Identify trends, correlations, and anomalies
- Feature Engineering – Encode categorical variables, scale numerical features
- Model Training – Random Forest with GridSearchCV for hyperparameter tuning
- Evaluation – Accuracy, Precision, Recall, F1‑score, ROC‑AUC
- Deployment – FastAPI backend + HTML/CSS frontend
- Testing – Unit tests with pytest
- CI/CD – Automated build, test, and deploy pipeline
- Containerization – Docker for consistent environments

📂 Project Structure
```
├── data/
│   ├── raw/                  # Original dataset(s)
│   ├── processed/            # Cleaned & transformed datasets
│
├── frontend/
│   ├── index.html            # Welcome page
│   ├── predict.html          # Loan prediction form
│   └── result.html           # Display prediction results
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_model_building.ipynb
│
├── scripts/
│   ├── preprocessing.py       # Clean and transform data
│   ├── train_model.py         # Train and return model
│   ├── predict.py             # Predict Loan Approval
│   ├── evaluate.py            # Evaluate and save results
│
├── models/                    # Saved ML models (.pkl)
│
├── test/
│   └── test_code.py           # pytest unit tests
│
├── results/
│   ├── training_testing_results.csv
│   ├── feature_importance.csv
│   └── best_parameter.csv
│
├── .dockerignore
├── Dockerfile
├── main.py                    # FastAPI app entry point
├── requirements.txt
├── README.md
└── .gitignore
```



## 🖥️ Frontend
- index.html → Welcome page with project intro
- predict.html → Form for users to input loan details (9 features)
- result.html → Displays prediction result (Approved/Rejected)

## ⚙️ Backend
- FastAPI app (main.py) exposes endpoints:
- / → Welcome page
- /predict → Loan approval prediction
- Uvicorn runs the ASGI server

## 🐳 Docker Support
A Dockerfile is included to containerize the app:
``` Dockerfile
from python:3.9-slim

# Set work directory
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . /app/

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```


Build and run:
```bash
docker build -t loan-approval-app .
docker run -p 8000:8000 loan-approval-app
```


## 🔄 CI/CD Pipeline
- Implemented with GitHub Actions
- Workflow includes:
- Install dependencies
- Run pytest
- Build Docker image
- Deploy to cloud (AWS, Azure, or GCP)
Example workflow file: .github/workflows/ci.yml
``` yml
name: CI Pipeline

on: [push, pull_request]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```


## 🧪 Testing
- pytest used for unit and integration tests
- Example test (test/test_code.py):
``` Python
from scripts.predict import predict_loan

def test_predict_loan():
    # Example input: [Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]
    result = predict_loan([1, "Graduate", "No", 5000, 2000, 150, 360, 1, "Urban"])
    assert result in [0, 1]   # 0 = Rejected, 1 = Approved
```

Run tests:
```bash
pytest
```


## 🚀 End-to-End Flow
- User opens frontend in browser
- Inputs loan application details into form
- Request sent to FastAPI backend
- Backend loads trained Random Forest model
- Model predicts loan approval (Approved/Rejected)
- Result displayed on result.html
- CI/CD ensures continuous testing and deployment
- Docker ensures reproducibility across environments

## ✅ Key Features
- End‑to‑end ML pipeline (EDA → Model → Deployment)
- Random Forest with GridSearchCV for hyperparameter tuning
- FastAPI backend with REST endpoints
- Frontend UI for user interaction
- Dockerized for portability
- CI/CD pipeline for automation
- Unit tests with pytest
