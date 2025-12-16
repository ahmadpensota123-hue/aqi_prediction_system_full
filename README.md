# 🌍 End-to-End Air Quality Index (AQI) Prediction System

> A production-grade Machine Learning system for predicting Air Quality Index (AQI) for the next 3 days using multiple ML models and industry-level MLOps practices.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📌 Project Overview

This project is developed for **AI321L – Machine Learning Lab** course, focusing on Earth & Environmental Intelligence domain.

### 🎯 Goals

- Predict AQI for the next 3 days
- Compare multiple ML models (Regression, Classification, Time Series)
- Implement full MLOps pipeline with CI/CD
- Deploy as production-ready API with dashboard

## 🏗️ System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│ Feature Store   │────▶│   ML Models     │
│  AQICN/OpenWx   │     │   (Hopsworks)   │     │ RF/XGB/LSTM/etc │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Dashboard     │◀────│   FastAPI       │◀────│ Model Registry  │
│   (Streamlit)   │     │   Service       │     │   (MLflow)      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Git
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/student/aqi-prediction-system.git
   cd aqi-prediction-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Getting API Keys

1. **AQICN API**: Get free key at [aqicn.org/data-platform/token/](https://aqicn.org/data-platform/token/)
2. **OpenWeather API**: Sign up at [openweathermap.org/api](https://openweathermap.org/api)
3. **Hopsworks**: Create free account at [app.hopsworks.ai](https://app.hopsworks.ai/)

## 📁 Project Structure

```
aqi-prediction-system/
├── src/                    # Source code
│   ├── config/             # Configuration management
│   ├── data/               # Data ingestion & feature engineering
│   ├── models/             # ML models (regression, classification, timeseries)
│   ├── evaluation/         # Model evaluation metrics
│   ├── feature_store/      # Hopsworks integration
│   └── utils/              # Utility functions
├── pipelines/              # Prefect workflow pipelines
├── api/                    # FastAPI application
├── dashboard/              # Streamlit dashboard
├── tests/                  # Unit, integration, ML validation tests
├── docker/                 # Docker configurations
├── .github/workflows/      # CI/CD pipelines
└── docs/                   # Documentation
```

## 🔧 Usage

### Run Feature Pipeline
```bash
python -m pipelines.feature_pipeline
```

### Run Training Pipeline
```bash
python -m pipelines.training_pipeline
```

### Start API Server
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Dashboard
```bash
streamlit run dashboard/app.py
```

### Run with Docker
```bash
docker-compose -f docker/docker-compose.yml up --build
```

## 📊 Models Implemented

| Model Type | Algorithm | Task |
|------------|-----------|------|
| Regression | Linear, Ridge, Lasso, Random Forest, XGBoost | AQI Value Prediction |
| Classification | Random Forest, XGBoost | AQI Category Prediction |
| Time Series | ARIMA, Prophet, LSTM | 3-Day Forecast |
| Neural Network | TensorFlow Dense/LSTM | Regression & Forecasting |

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/predict` | POST | Real-time AQI prediction |
| `/forecast` | POST | 3-day AQI forecast |
| `/docs` | GET | Swagger UI documentation |

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test types
pytest tests/unit/ -v           # Unit tests
pytest tests/integration/ -v    # Integration tests
pytest tests/ml_validation/ -v  # ML validation tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows for:
- ✅ Code quality checks (Black, isort, flake8)
- ✅ Unit and integration tests
- ✅ ML validation tests (DeepChecks)
- ✅ Docker image building
- ✅ Automated deployment

## 📝 Documentation

- [Project Report](docs/project_report.md)
- [API Documentation](docs/api_documentation.md)
- [Architecture Guide](docs/architecture.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Course: AI321L – Machine Learning Lab
- Domain: Earth & Environmental Intelligence
- Data Sources: AQICN, OpenWeatherMap
