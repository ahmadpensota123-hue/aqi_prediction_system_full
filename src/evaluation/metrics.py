"""
Model Evaluation Metrics
========================

This module provides comprehensive evaluation metrics for AQI prediction models.

Metrics Implemented:
- Regression: RMSE, MAE, MAPE, R², Adjusted R²
- Classification: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Time Series: RMSE, MAE, MASE (Mean Absolute Scaled Error)

Why Multiple Metrics?
- Different metrics capture different aspects of model performance
- RMSE penalizes large errors more than MAE
- F1-Score balances precision and recall
- R² shows variance explained
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

from sklearn.metrics import (
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from src.config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCalculator:
    """
    Calculator for various ML evaluation metrics.
    
    Provides static methods for calculating metrics consistently
    across all models in the system.
    """
    
    # ========================
    # REGRESSION METRICS
    # ========================
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.
        
        RMSE penalizes larger errors more heavily, making it
        sensitive to outliers. Good when large errors are costly.
        
        Formula: sqrt(mean((y_true - y_pred)²))
        
        Interpretation:
        - Lower is better
        - Same units as target variable
        - RMSE of 10 means predictions off by ~10 AQI units on average
        """
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        MAE treats all errors equally regardless of direction.
        More robust to outliers than RMSE.
        
        Formula: mean(|y_true - y_pred|)
        
        Interpretation:
        - Lower is better
        - Same units as target variable
        - MAE of 5 means predictions off by 5 AQI units on average
        """
        return float(mean_absolute_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        
        MAPE expresses error as a percentage, making it easier
        to interpret across different scales.
        
        Formula: mean(|y_true - y_pred| / |y_true|) * 100
        
        Note: Undefined when y_true contains zeros.
        """
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return float("nan")
        
        return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R-squared (Coefficient of Determination).
        
        Measures proportion of variance in target explained by model.
        
        Formula: 1 - (SS_res / SS_tot)
        
        Interpretation:
        - R² = 1.0: Perfect predictions
        - R² = 0.0: Model predicts mean of target
        - R² < 0.0: Model worse than predicting mean
        """
        return float(r2_score(y_true, y_pred))
    
    @staticmethod
    def adjusted_r2(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int
    ) -> float:
        """
        Adjusted R-squared.
        
        Adjusts R² for the number of features, penalizing
        model complexity. Better for model comparison.
        
        Formula: 1 - (1 - R²) * (n - 1) / (n - p - 1)
        where n = samples, p = features
        """
        n = len(y_true)
        r2_val = r2_score(y_true, y_pred)
        
        # Avoid division by zero
        if n - n_features - 1 <= 0:
            return float("nan")
        
        adj_r2 = 1 - (1 - r2_val) * (n - 1) / (n - n_features - 1)
        return float(adj_r2)
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int = None
    ) -> Dict[str, float]:
        """
        Calculate all regression metrics.
        
        Returns dictionary with all metrics for easy comparison.
        """
        metrics = {
            "rmse": MetricsCalculator.rmse(y_true, y_pred),
            "mae": MetricsCalculator.mae(y_true, y_pred),
            "mape": MetricsCalculator.mape(y_true, y_pred),
            "r2": MetricsCalculator.r2(y_true, y_pred),
        }
        
        if n_features is not None:
            metrics["adjusted_r2"] = MetricsCalculator.adjusted_r2(
                y_true, y_pred, n_features
            )
        
        return metrics
    
    # ========================
    # CLASSIFICATION METRICS
    # ========================
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Classification Accuracy.
        
        Proportion of correct predictions.
        
        Warning: Can be misleading for imbalanced classes.
        AQI categories are often imbalanced (more "Good" days than "Hazardous").
        """
        return float(accuracy_score(y_true, y_pred))
    
    @staticmethod
    def precision(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted"
    ) -> float:
        """
        Precision (Positive Predictive Value).
        
        Of all positive predictions, how many were correct?
        
        High precision = few false positives
        Important when false alarms are costly.
        """
        return float(precision_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def recall(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted"
    ) -> float:
        """
        Recall (Sensitivity, True Positive Rate).
        
        Of all actual positives, how many did we catch?
        
        High recall = few false negatives
        Important for AQI - we don't want to miss unhealthy days!
        """
        return float(recall_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def f1(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        average: str = "weighted"
    ) -> float:
        """
        F1-Score.
        
        Harmonic mean of precision and recall.
        Balance between precision and recall.
        
        Averages:
        - "weighted": Account for class imbalance
        - "macro": Simple average across classes
        - "micro": Aggregate TP, FP, FN across classes
        """
        return float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all classification metrics."""
        return {
            "accuracy": MetricsCalculator.accuracy(y_true, y_pred),
            "precision_weighted": MetricsCalculator.precision(y_true, y_pred, "weighted"),
            "precision_macro": MetricsCalculator.precision(y_true, y_pred, "macro"),
            "recall_weighted": MetricsCalculator.recall(y_true, y_pred, "weighted"),
            "recall_macro": MetricsCalculator.recall(y_true, y_pred, "macro"),
            "f1_weighted": MetricsCalculator.f1(y_true, y_pred, "weighted"),
            "f1_macro": MetricsCalculator.f1(y_true, y_pred, "macro"),
        }
    
    # ========================
    # TIME SERIES METRICS
    # ========================
    
    @staticmethod
    def mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray = None,
        seasonality: int = 1
    ) -> float:
        """
        Mean Absolute Scaled Error.
        
        MASE is scale-independent and compares forecast error
        to naive forecast (using previous value or seasonal value).
        
        MASE < 1: Better than naive forecast
        MASE > 1: Worse than naive forecast
        """
        if y_train is None:
            y_train = y_true
        
        # Calculate naive forecast error
        naive_errors = np.abs(np.diff(y_train, n=seasonality))
        scale = np.mean(naive_errors)
        
        if scale == 0:
            return float("nan")
        
        # Calculate prediction errors
        errors = np.abs(y_true - y_pred)
        
        return float(np.mean(errors) / scale)


class ModelEvaluator:
    """
    High-level model evaluation and comparison.
    
    Provides utilities for:
    - Evaluating multiple models
    - Generating comparison reports
    - Saving evaluation results
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.results_dir = Path("data/evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_history = []
    
    def evaluate_regression_models(
        self,
        models_dict: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate multiple regression models.
        
        Args:
            models_dict: Dictionary mapping model names to (model, predict_func)
            X: Feature DataFrame
            y: True target values
        
        Returns:
            DataFrame with metrics for each model
        """
        results = []
        
        for name, model_info in models_dict.items():
            if isinstance(model_info, tuple):
                model, predict_func = model_info
                y_pred = predict_func(X)
            else:
                model = model_info
                y_pred = model.predict(X)
            
            metrics = MetricsCalculator.calculate_regression_metrics(
                y.values, y_pred, n_features=X.shape[1]
            )
            metrics["model"] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df[["model", "rmse", "mae", "mape", "r2", "adjusted_r2"]]
        
        return df.sort_values("rmse")
    
    def evaluate_classification_models(
        self,
        models_dict: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate multiple classification models.
        """
        results = []
        
        for name, model_info in models_dict.items():
            if isinstance(model_info, tuple):
                model, predict_func = model_info
                y_pred = predict_func(X)
            else:
                model = model_info
                y_pred = model.predict(X)
            
            metrics = MetricsCalculator.calculate_classification_metrics(y.values, y_pred)
            metrics["model"] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df[["model", "accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]]
        
        return df.sort_values("f1_weighted", ascending=False)
    
    def generate_comparison_report(
        self,
        regression_results: pd.DataFrame = None,
        classification_results: pd.DataFrame = None,
        timeseries_results: pd.DataFrame = None
    ) -> str:
        """
        Generate a text report comparing all models.
        """
        report_lines = [
            "=" * 60,
            "MODEL COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]
        
        if regression_results is not None and len(regression_results) > 0:
            report_lines.append("REGRESSION MODELS")
            report_lines.append("-" * 40)
            report_lines.append(regression_results.to_string(index=False))
            report_lines.append("")
            
            best = regression_results.iloc[0]
            report_lines.append(f"Best Model: {best['model']} (RMSE: {best['rmse']:.2f})")
            report_lines.append("")
        
        if classification_results is not None and len(classification_results) > 0:
            report_lines.append("CLASSIFICATION MODELS")
            report_lines.append("-" * 40)
            report_lines.append(classification_results.to_string(index=False))
            report_lines.append("")
            
            best = classification_results.iloc[0]
            report_lines.append(f"Best Model: {best['model']} (F1: {best['f1_weighted']:.3f})")
            report_lines.append("")
        
        if timeseries_results is not None and len(timeseries_results) > 0:
            report_lines.append("TIME SERIES MODELS")
            report_lines.append("-" * 40)
            report_lines.append(timeseries_results.to_string(index=False))
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_evaluation(
        self,
        results: Dict[str, Any],
        filename: str = None
    ) -> Path:
        """
        Save evaluation results to JSON.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
        
        filepath = self.results_dir / filename
        
        # Convert DataFrames to dicts
        save_data = {
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                save_data["results"][key] = value.to_dict(orient="records")
            else:
                save_data["results"][key] = value
        
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved evaluation to {filepath}")
        
        return filepath


if __name__ == "__main__":
    # Test metrics
    print("Testing Evaluation Metrics")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.uniform(50, 200, n_samples)
    y_pred = y_true + np.random.normal(0, 20, n_samples)
    
    print("\n1. Regression Metrics:")
    metrics = MetricsCalculator.calculate_regression_metrics(y_true, y_pred, n_features=10)
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Classification
    print("\n2. Classification Metrics:")
    y_true_class = np.random.randint(0, 3, n_samples)
    y_pred_class = y_true_class.copy()
    y_pred_class[::10] = (y_pred_class[::10] + 1) % 3  # Add some errors
    
    class_metrics = MetricsCalculator.calculate_classification_metrics(y_true_class, y_pred_class)
    for name, value in class_metrics.items():
        print(f"   {name}: {value:.4f}")
    
    # Time series
    print("\n3. Time Series Metrics:")
    mase = MetricsCalculator.mase(y_true, y_pred)
    print(f"   MASE: {mase:.4f}")
    
    print("\n" + "=" * 50)
    print("Evaluation Metrics Test Complete!")
