# Telco Customer Churn Prediction

An end-to-end MLOps project that predicts customer churn for a telecommunications company. Built with XGBoost, FastAPI, Gradio, MLflow, and deployed on AWS ECS Fargate.

**Live Demo**: [Application URL via AWS ALB]

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Training](#model-training)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Challenges and Solutions](#challenges-and-solutions)

---

## Overview

### The Problem

Customer churn is a critical business metric for telecom companies. Identifying at-risk customers before they leave allows proactive retention efforts, reducing revenue loss.

### The Solution

This project delivers a complete ML pipeline that:

- **Predicts churn** using an XGBoost classifier trained on historical customer data
- **Serves predictions** via a REST API and interactive web UI
- **Tracks experiments** with MLflow for reproducibility
- **Deploys automatically** to AWS using CI/CD pipelines

### Key Metrics

| Metric    | Value |
|-----------|-------|
| ROC AUC   | ~0.85 |
| Precision | ~0.65 |
| Recall    | ~0.78 |
| F1 Score  | ~0.71 |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| ML Framework | XGBoost, scikit-learn, pandas, numpy |
| Data Validation | Great Expectations |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, uvicorn |
| Web UI | Gradio |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | AWS ECS Fargate, Application Load Balancer, CloudWatch |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Raw Data → Data Validation → Preprocessing → Feature Engineering       │
│      │           │                                    │                  │
│      │    Great Expectations                   Binary + One-Hot         │
│      │                                         Encoding                  │
│      ▼                                                │                  │
│  CSV File                                             ▼                  │
│                                              XGBoost Training            │
│                                                       │                  │
│                                                       ▼                  │
│                                               MLflow Logging             │
│                                          (metrics, params, artifacts)    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           SERVING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐           │
│    │   FastAPI    │     │   Feature    │     │   XGBoost    │           │
│    │  POST /predict ──▶│  Transform   │────▶│    Model     │           │
│    └──────────────┘     └──────────────┘     └──────────────┘           │
│           │                                         │                    │
│           │                                         ▼                    │
│    ┌──────────────┐                         ┌──────────────┐            │
│    │  Gradio UI   │                         │  Prediction  │            │
│    │    /ui       │                         │   Response   │            │
│    └──────────────┘                         └──────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   GitHub Push → GitHub Actions → Docker Hub → ECS Service Update        │
│       │              │               │              │                    │
│       │         Build Image     Push Image    Force Deploy              │
│       │                                             │                    │
│       │                                             ▼                    │
│       │                                    ┌───────────────┐            │
│       │                                    │  AWS Fargate  │            │
│       │                                    │    Task       │            │
│       │                                    └───────┬───────┘            │
│       │                                            │                    │
│       │                                            ▼                    │
│       │                                    ┌───────────────┐            │
│       │                                    │     ALB       │            │
│       └───────────────────────────────────▶│  HTTP:80      │            │
│                                            └───────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
telco-customer-churn-ml/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned dataset
├── src/
│   ├── app/
│   │   └── main.py             # FastAPI + Gradio application
│   ├── data/
│   │   ├── load_data.py        # Data loading utilities
│   │   └── preprocess.py       # Data cleaning
│   ├── features/
│   │   └── build_features.py   # Feature engineering
│   ├── models/
│   │   ├── train.py            # Model training
│   │   ├── evaluate.py         # Model evaluation
│   │   └── tune.py             # Hyperparameter tuning
│   ├── serving/
│   │   ├── inference.py        # Prediction logic
│   │   └── model/              # Serialized model artifacts
│   └── utils/
│       ├── utils.py            # Helper functions
│       └── validate_data.py    # Great Expectations validation
├── scripts/
│   ├── run_pipeline.py         # Main training pipeline
│   ├── prepare_processed_data.py
│   └── test_*.py               # Test scripts
├── notebooks/
│   └── EDA.ipynb               # Exploratory data analysis
├── mlruns/                     # MLflow experiment tracking
├── artifacts/                  # Shared preprocessing artifacts
├── .github/workflows/
│   └── ci.yml                  # CI/CD pipeline
├── dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- AWS CLI (for cloud deployment)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/telco-customer-churn-ml.git
cd telco-customer-churn-ml
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Application Locally

```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Access the application:
- **API Documentation**: http://localhost:8000/docs
- **Web UI**: http://localhost:8000/ui
- **Health Check**: http://localhost:8000/

### Run with Docker

```bash
# Build the image
docker build -t telco-churn-app .

# Run the container
docker run -p 8000:8000 telco-churn-app
```

---

## Model Training

### Run the Training Pipeline

```bash
python scripts/run_pipeline.py \
    --input data/raw/Telco-Customer-Churn.csv \
    --target Churn
```

### Pipeline Stages

1. **Data Loading** - Load raw CSV data
2. **Data Validation** - Validate data quality with Great Expectations
3. **Preprocessing** - Handle missing values, fix data types
4. **Feature Engineering** - Binary encoding + one-hot encoding
5. **Model Training** - Train XGBoost with optimized hyperparameters
6. **Evaluation** - Calculate precision, recall, F1, ROC AUC
7. **Artifact Logging** - Save model and metadata to MLflow

### View Experiment Results

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open http://localhost:5000 to view experiment runs, metrics, and artifacts.

### XGBoost Configuration

The model uses optimized hyperparameters:

```python
XGBClassifier(
    n_estimators=301,
    learning_rate=0.034,
    max_depth=7,
    subsample=0.95,
    colsample_bytree=0.98,
    scale_pos_weight=calculated_dynamically,  # Handles class imbalance
)
```

---

## API Reference

### Health Check

```
GET /
```

Response:
```json
{"status": "ok"}
```

### Predict Churn

```
POST /predict
```

Request body:
```json
{
  "gender": "Female",
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure": 1,
  "MonthlyCharges": 85.0,
  "TotalCharges": 85.0
}
```

Response:
```json
{"prediction": "Likely to churn"}
```

### Example with cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "Yes",
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)",
    "tenure": 60,
    "MonthlyCharges": 45.0,
    "TotalCharges": 2700.0
  }'
```

---

## Deployment

### CI/CD Pipeline

The GitHub Actions workflow automatically builds and pushes Docker images on every push to `main`:

```yaml
# .github/workflows/ci.yml
on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: your-username/telco-fastapi:latest
```

### Required GitHub Secrets

Add these secrets in your repository settings:

| Secret | Description |
|--------|-------------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

### AWS Infrastructure

The application runs on AWS ECS Fargate with the following setup:

- **ECS Cluster**: Serverless container orchestration
- **ECS Service**: Maintains desired task count
- **Task Definition**: Container configuration (port 8000)
- **Application Load Balancer**: Routes HTTP:80 to container:8000
- **Security Groups**: ALB allows inbound 80, task allows inbound 8000 from ALB
- **CloudWatch Logs**: Container stdout/stderr logging

### Deploy to ECS

After the CI/CD pipeline pushes a new image:

```bash
# Force new deployment to pull latest image
aws ecs update-service \
  --cluster your-cluster-name \
  --service your-service-name \
  --force-new-deployment
```

---

## Challenges and Solutions

### 1. Unhealthy Targets Behind ALB

**Problem**: Application Load Balancer showed targets as unhealthy.

**Root Cause**: App didn't respond at the health-check path; port mismatches between listener and target group.

**Solution**: 
- Added `GET /` health endpoint returning `{"status": "ok"}`
- Configured ALB listener on port 80 to forward to target group on port 8000
- Set target group health check path to `/`

### 2. Module Import Error in Container

**Problem**: `ModuleNotFoundError: serving` when running in Docker.

**Root Cause**: Python path in the container didn't include the `src/` directory.

**Solution**: Set `PYTHONPATH=/app/src` in the Dockerfile.

### 3. Training/Serving Feature Mismatch

**Problem**: Model predictions were inconsistent between training and serving.

**Root Cause**: Feature transformations differed between training and inference pipelines.

**Solution**: 
- Created fixed `BINARY_MAP` dictionary in inference code matching training logic
- Saved feature column order to `feature_columns.txt` during training
- Inference pipeline reindexes features to match exact training order

### 4. Class Imbalance

**Problem**: Model biased toward predicting non-churn due to imbalanced dataset.

**Root Cause**: ~73% of customers don't churn, leading to biased predictions.

**Solution**: 
- Calculated `scale_pos_weight` dynamically based on class ratio
- Applied to XGBoost to give more weight to the minority class (churners)
- Used classification threshold of 0.35 instead of default 0.5

---

## Testing

```bash
# Test data processing and feature engineering
python scripts/test_pipeline_phase1_data_features.py

# Test model training and evaluation
python scripts/test_pipeline_phase2_modeling.py

# Test FastAPI endpoints
python scripts/test_fastapi.py
```

---

## License

MIT

---

## Author

Built as an end-to-end MLOps portfolio project demonstrating the complete machine learning lifecycle from data processing to cloud deployment.
