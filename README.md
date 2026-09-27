# 🫁 End-to-End Chest Cancer Image Classification

An end-to-end Deep Learning system for **Chest Cancer Detection and CT Scan Classification** built with **TensorFlow (VGG16 Transfer Learning)**, **MLflow** for experiment tracking, **DVC** for data pipeline reproducibility, **Flask** for serving predictions through an interactive web app, **Docker** for containerization, and **GitHub Actions** for continuous deployment to **AWS (ECR & EC2)**.

---

## 🌟 Features

- 🔬 **Multi-Class CT Scan Classification**: Classifies chest scans into 4 clinical categories:
  1. **Adenocarcinoma**
  2. **Large Cell Carcinoma**
  3. **Squamous Cell Carcinoma**
  4. **Normal (Non-Cancerous)**
- ⚙️ **Modular Config-Driven Architecture**: Managed via `config/config.yaml` for paths and `params.yaml` for hyperparameters.
- 🔁 **Reproducible Pipelines (DVC)**: Track dependencies, inputs, outputs, and metrics for each stage with `dvc.yaml`.
- 📊 **Experiment Tracking (MLflow)**: Log parameters, loss/accuracy metrics, confusion matrices, and model checkpoints to MLflow.
- 💻 **Flask Web Application**: Modern UI featuring drag-and-drop file uploads, live sample CT scan selectors, dynamic probability breakdown bars, risk badges, and diagnostic summaries.
- 🐳 **Docker & Docker Compose**: Full containerization for reproducible local and cloud execution.
- 🚀 **AWS Deployment Pipeline**: Automated CI/CD workflow via GitHub Actions pushing Docker images to Amazon ECR and deploying to AWS EC2.

---

## 🏗️ Project Architecture & Pipeline Stages

```
End-to-End-Chest-Cancer-Classification/
├── .github/workflows/main.yaml   # GitHub Actions CI/CD workflow for AWS
├── config/config.yaml            # Paths configuration
├── params.yaml                   # Model hyperparameters
├── dvc.yaml                      # DVC pipeline stages definition
├── Dockerfile                    # Docker build configuration
├── docker-compose.yml            # Docker compose configuration
├── app.py                        # Flask Web Application entry point
├── main.py                       # Execution pipeline script
├── requirements.txt              # Dependencies
├── setup.py                      # Package configuration
├── static/                       # CSS, JavaScript & sample images
├── templates/index.html          # HTML UI template
└── src/chestCancerClassifier/
    ├── components/               # Data ingestion, base model, trainer & evaluator
    ├── config/                   # Configuration manager
    ├── entity/                   # Dataclasses
    ├── pipeline/                 # Stage execution scripts & predictor
    └── utils/                    # Common helper utilities
```

### Modular Pipeline Stages:
1. **Stage 01: Data Ingestion** (`src/chestCancerClassifier/pipeline/stage_01_data_ingestion.py`)
   - Downloads/extracts CT scan dataset into `artifacts/data_ingestion/`.
2. **Stage 02: Prepare Base Model** (`src/chestCancerClassifier/pipeline/stage_02_prepare_base_model.py`)
   - Prepares pretrained VGG16 backbone with custom dense classification head.
3. **Stage 03: Model Training** (`src/chestCancerClassifier/pipeline/stage_03_model_trainer.py`)
   - Trains model with data augmentation, image generators, and saves `artifacts/training/model.h5`.
4. **Stage 04: Model Evaluation** (`src/chestCancerClassifier/pipeline/stage_04_model_evaluation.py`)
   - Evaluates test accuracy/loss, logs parameters & artifacts to MLflow, and exports `metrics.json`.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/ml-engineer/End-to-End-Chest-Cancer-Classification.git
cd End-to-End-Chest-Cancer-Classification

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate
# Activate environment (Linux/macOS)
source venv/bin/activate

# Install dependencies and local package
pip install -r requirements.txt
```

---

### 2. Run Training Pipeline

#### Option A: Direct Python Execution
Run all pipeline stages sequentially:
```bash
python main.py
```

#### Option B: DVC Pipeline Reproduction
Run data pipeline using DVC to track stage dependencies:
```bash
dvc repro
```
To view tracked metrics:
```bash
dvc metrics show
```

---

### 3. Experiment Tracking with MLflow

Launch local MLflow UI to inspect logged hyperparameter runs, loss curves, and model metrics:
```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

### 4. Run Flask Web App Locally

Launch the Flask server:
```bash
python app.py
```
Open [http://localhost:8080](http://localhost:8080) to interact with the application.

#### Available Endpoints:
- `GET /`: Main web interface.
- `GET /health`: Server health check (`{"status": "healthy"}`).
- `POST /predict`: Submit image file or sample path for classification.
- `POST /train`: Trigger training pipeline run.

---

### 5. Run with Docker & Docker Compose

#### Build & Run Container directly:
```bash
# Build image
docker build -t chest-cancer-app .

# Run container
docker run -p 8080:8080 chest-cancer-app
```

#### Run with Docker Compose:
```bash
docker-compose up --build
```

---

## ☁️ Deployment Guide (AWS ECR + EC2 via GitHub Actions)

### Step 1: AWS Setup
1. **Create an AWS ECR Repository**: Note the URI (`<aws_account_id>.dkr.ecr.<region>.amazonaws.com/<repo_name>`).
2. **Launch an AWS EC2 Instance** (Ubuntu):
   - Configure Security Group: Allow inbound traffic on port `8080` and `22` (SSH).
   - Install Docker on EC2:
     ```bash
     sudo apt-get update -y
     sudo apt-get install -y docker.io
     sudo systemctl start docker
     sudo systemctl enable docker
     sudo usermod -aG docker ubuntu
     newgrp docker
     ```

### Step 2: Configure Self-Hosted GitHub Actions Runner on EC2
1. Go to your GitHub Repository -> **Settings** -> **Actions** -> **Runners** -> **New self-hosted runner**.
2. Follow the OS commands on your EC2 instance terminal to configure and start the runner background service (`./run.sh` or `./externals/node16/bin/node ./bin/runnerService.js`).

### Step 3: Add Secrets to GitHub Repository
Go to **Settings** -> **Secrets and variables** -> **Actions** and add:

| Secret Name | Description |
| :--- | :--- |
| `AWS_ACCESS_KEY_ID` | AWS IAM User Access Key |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM User Secret Key |
| `AWS_REGION` | AWS Region (e.g., `us-east-1`) |
| `AWS_ECR_LOGIN_URI` | ECR Registry URI (`<account_id>.dkr.ecr.<region>.amazonaws.com`) |
| `ECR_REPOSITORY_NAME` | ECR Repository Name (`chest-cancer-classifier`) |

### Step 4: Trigger Deployment
Push changes to the `main` branch to automatically trigger the `.github/workflows/main.yaml` CI/CD pipeline!

---

## 📜 License

This project is open-source and available under the MIT License.
