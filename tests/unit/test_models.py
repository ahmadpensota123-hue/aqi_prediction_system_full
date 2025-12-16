"""
Unit Tests for Models Module
============================
"""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.feature_engineering import generate_synthetic_training_data, FeatureEngineer
from src.models.regression import RegressionModels
from src.models.classification import ClassificationModels
from src.evaluation.metrics import MetricsCalculator


class TestRegressionModels:
    """Tests for regression models."""
    
    @pytest.fixture
    def training_data(self):
        """Create training data."""
        df = generate_synthetic_training_data(n_samples=200)
        fe = FeatureEngineer()
        df = fe.add_lag_features(df)
        df = fe.add_rolling_features(df)
        df = fe.add_derived_features(df)
        df = df.dropna()
        
        exclude = ["timestamp", "city", "aqi_category_name", "aqi_category",
                   "dominant_pollutant", "weather_condition"]
        X = df[[c for c in df.columns if c not in exclude]]
        y = df["aqi"]
        
        return X, y
    
    def test_model_initialization(self):
        """Test model initialization."""
        models = RegressionModels(models_to_use=["linear", "ridge"])
        
        assert "linear" in models.models
        assert "ridge" in models.models
        assert len(models.models) == 2
    
    def test_model_training(self, training_data):
        """Test model training."""
        X, y = training_data
        models = RegressionModels(models_to_use=["linear", "ridge"])
        results = models.train(X, y)
        
        assert "linear" in results
        assert "ridge" in results
        assert "val_rmse" in results["linear"]
    
    def test_model_prediction(self, training_data):
        """Test model prediction."""
        X, y = training_data
        models = RegressionModels(models_to_use=["linear"])
        models.train(X, y)
        
        predictions = models.predict(X.head(10), model_name="linear")
        
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestClassificationModels:
    """Tests for classification models."""
    
    @pytest.fixture
    def training_data(self):
        """Create training data."""
        df = generate_synthetic_training_data(n_samples=200)
        fe = FeatureEngineer()
        df = fe.add_lag_features(df)
        df = df.dropna()
        
        exclude = ["timestamp", "city", "aqi_category_name", "aqi_category",
                   "dominant_pollutant", "weather_condition"]
        X = df[[c for c in df.columns if c not in exclude]]
        y = df["aqi_category_name"]
        
        return X, y
    
    def test_classification_training(self, training_data):
        """Test classification model training."""
        X, y = training_data
        models = ClassificationModels(models_to_use=["logistic"])
        results = models.train(X, y)
        
        assert "logistic" in results
        assert "val_accuracy" in results["logistic"]


class TestMetricsCalculator:
    """Tests for evaluation metrics."""
    
    def test_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        rmse = MetricsCalculator.rmse(y_true, y_pred)
        
        assert rmse > 0
        assert rmse < 0.2
    
    def test_mae(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mae = MetricsCalculator.mae(y_true, y_pred)
        
        assert mae == 0
    
    def test_r2(self):
        """Test R² calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = MetricsCalculator.r2(y_true, y_pred)
        
        assert r2 == 1.0
    
    def test_accuracy(self):
        """Test accuracy calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        
        acc = MetricsCalculator.accuracy(y_true, y_pred)
        
        assert acc == 1.0
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        f1 = MetricsCalculator.f1(y_true, y_pred, average="weighted")
        
        assert 0 <= f1 <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
