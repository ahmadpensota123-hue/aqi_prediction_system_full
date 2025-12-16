"""
Training Pipeline - Prefect Workflow
====================================

This pipeline handles automated model training and evaluation.

Pipeline Steps:
1. Load training data from Feature Store
2. Train multiple ML models (Regression, Classification, Time Series)
3. Evaluate models and compare performance
4. Save best models to Model Registry
5. Send notifications with results

Scheduling:
- Runs daily to retrain on new data
- Can be triggered manually for experiments
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import traceback
import json

from prefect import flow, task, get_run_logger

# Import our modules
from src.config.settings import get_settings
from src.data.feature_engineering import FeatureEngineer, generate_synthetic_training_data
from src.feature_store.hopsworks_client import FeatureStoreManager
from src.models.regression import RegressionModels
from src.models.classification import ClassificationModels
from src.models.timeseries import TimeSeriesModels
from src.evaluation.metrics import MetricsCalculator, ModelEvaluator


@task(
    name="Load Training Data",
    description="Load features from Feature Store",
    retries=2,
    retry_delay_seconds=30
)
def load_training_data(
    use_synthetic: bool = False,
    n_samples: int = 1000
) -> Any:
    """
    Task: Load training data from Feature Store.
    
    Falls back to synthetic data if real data not available.
    
    Args:
        use_synthetic: Force use of synthetic data for testing
        n_samples: Number of synthetic samples if used
    
    Returns:
        Training DataFrame
    """
    logger = get_run_logger()
    
    if use_synthetic:
        logger.info(f"Generating {n_samples} synthetic samples")
        df = generate_synthetic_training_data(n_samples=n_samples)
        return df
    
    # Try to load from feature store
    try:
        fs_manager = FeatureStoreManager()
        fs_manager.initialize()
        df = fs_manager.get_training_data("aqi_features", days_back=30)
        fs_manager.close()
        
        if len(df) > 0:
            logger.info(f"Loaded {len(df)} samples from Feature Store")
            return df
    except Exception as e:
        logger.warning(f"Could not load from Feature Store: {e}")
    
    # Fall back to synthetic
    logger.info("Falling back to synthetic data")
    return generate_synthetic_training_data(n_samples=n_samples)


@task(
    name="Prepare Features",
    description="Prepare features for model training"
)
def prepare_features(
    df: Any,
    target_column: str = "aqi",
    forecast_horizon: int = 1
) -> Dict[str, Any]:
    """
    Task: Prepare feature matrix and target vector.
    
    Returns dict with X_train, X_test, y_train, y_test.
    """
    logger = get_run_logger()
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    fe = FeatureEngineer()
    
    # Add engineered features
    df = fe.add_lag_features(df)
    df = fe.add_rolling_features(df)
    df = fe.add_derived_features(df)
    df = df.dropna()
    
    # Prepare X and y
    exclude_cols = [
        "timestamp", "city", "aqi_category_name", "aqi_category",
        "dominant_pollutant", "weather_condition"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols]
    y_regression = df["aqi"]
    y_classification = df["aqi_category_name"] if "aqi_category_name" in df.columns else df["aqi_category"]
    
    # Train-test split
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    _, _, y_cls_train, y_cls_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )
    
    logger.info(f"Prepared data: Train={len(X_train)}, Test={len(X_test)}")
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_regression_train": y_reg_train,
        "y_regression_test": y_reg_test,
        "y_classification_train": y_cls_train,
        "y_classification_test": y_cls_test,
        "full_df": df
    }


@task(
    name="Train Regression Models",
    description="Train regression models for AQI prediction"
)
def train_regression(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task: Train regression models.
    
    Returns trained models and results.
    """
    logger = get_run_logger()
    
    models = RegressionModels()
    
    # Combine train sets for full training
    results = models.train(
        data["X_train"],
        data["y_regression_train"],
        validation_split=0.2
    )
    
    # Evaluate on test set
    test_metrics = models.evaluate(
        data["X_test"],
        data["y_regression_test"]
    )
    
    # Get best model
    best_name, best_model = models.get_best_model("rmse")
    
    logger.info(f"Best regression model: {best_name}")
    for name, metrics in test_metrics.items():
        logger.info(f"  {name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
    
    return {
        "models": models,
        "train_results": results,
        "test_metrics": test_metrics,
        "best_model": best_name
    }


@task(
    name="Train Classification Models",
    description="Train classification models for AQI category prediction"
)
def train_classification(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task: Train classification models.
    """
    logger = get_run_logger()
    
    models = ClassificationModels()
    
    results = models.train(
        data["X_train"],
        data["y_classification_train"],
        validation_split=0.2
    )
    
    test_metrics = models.evaluate(
        data["X_test"],
        data["y_classification_test"]
    )
    
    best_name, best_model = models.get_best_model()
    
    logger.info(f"Best classification model: {best_name}")
    
    return {
        "models": models,
        "train_results": results,
        "test_metrics": test_metrics,
        "best_model": best_name
    }


@task(
    name="Train Time Series Models",
    description="Train time series models for AQI forecasting"
)
def train_timeseries(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task: Train time series models.
    """
    logger = get_run_logger()
    
    df = data["full_df"]
    
    models = TimeSeriesModels(forecast_horizon=72)
    results = {}
    
    # Train Prophet
    try:
        prophet_result = models.train_prophet(df, "aqi")
        results["prophet"] = prophet_result
        logger.info(f"Prophet RMSE: {prophet_result.get('rmse', 'N/A')}")
    except Exception as e:
        logger.warning(f"Prophet training failed: {e}")
        results["prophet"] = {"error": str(e)}
    
    # Train LSTM (if TensorFlow available)
    try:
        lstm_result = models.train_lstm(df, "aqi", epochs=10)
        results["lstm"] = lstm_result
        logger.info(f"LSTM RMSE: {lstm_result.get('rmse', 'N/A')}")
    except Exception as e:
        logger.warning(f"LSTM training failed: {e}")
        results["lstm"] = {"error": str(e)}
    
    return {
        "models": models,
        "results": results
    }


@task(
    name="Save Models",
    description="Save trained models to Model Registry"
)
def save_models(
    regression_results: Dict,
    classification_results: Dict,
    timeseries_results: Dict
) -> Dict[str, str]:
    """
    Task: Save all trained models.
    """
    logger = get_run_logger()
    
    saved_paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save regression models
    if "models" in regression_results:
        path = regression_results["models"].save(f"regression_{timestamp}")
        saved_paths["regression"] = str(path)
        logger.info(f"Saved regression models: {path}")
    
    # Save classification models
    if "models" in classification_results:
        path = classification_results["models"].save(f"classification_{timestamp}")
        saved_paths["classification"] = str(path)
        logger.info(f"Saved classification models: {path}")
    
    # Save time series models
    if "models" in timeseries_results:
        try:
            path = timeseries_results["models"].save(f"timeseries_{timestamp}")
            saved_paths["timeseries"] = str(path)
            logger.info(f"Saved time series models: {path}")
        except Exception as e:
            logger.warning(f"Could not save time series models: {e}")
    
    return saved_paths


@task(
    name="Generate Report",
    description="Generate training report"
)
def generate_report(
    regression_results: Dict,
    classification_results: Dict,
    timeseries_results: Dict
) -> str:
    """
    Task: Generate training report.
    """
    logger = get_run_logger()
    
    import pandas as pd
    
    lines = [
        "=" * 60,
        "MODEL TRAINING REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]
    
    # Regression results
    lines.append("REGRESSION MODELS (Test Set)")
    lines.append("-" * 40)
    if "test_metrics" in regression_results:
        for name, metrics in regression_results["test_metrics"].items():
            lines.append(f"  {name}: RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
        lines.append(f"  Best: {regression_results.get('best_model', 'N/A')}")
    lines.append("")
    
    # Classification results
    lines.append("CLASSIFICATION MODELS (Test Set)")
    lines.append("-" * 40)
    if "test_metrics" in classification_results:
        for name, metrics in classification_results["test_metrics"].items():
            lines.append(f"  {name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_weighted']:.3f}")
        lines.append(f"  Best: {classification_results.get('best_model', 'N/A')}")
    lines.append("")
    
    # Time series results
    lines.append("TIME SERIES MODELS")
    lines.append("-" * 40)
    if "results" in timeseries_results:
        for name, result in timeseries_results["results"].items():
            if "error" not in result:
                lines.append(f"  {name.upper()}: RMSE={result.get('rmse', 'N/A'):.2f}")
            else:
                lines.append(f"  {name.upper()}: Error - {result['error'][:50]}")
    lines.append("")
    lines.append("=" * 60)
    
    report = "\n".join(lines)
    
    # Save report
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    logger.info(f"Report saved to {report_path}")
    
    return report


@task(
    name="Send Training Notification",
    description="Send notification about training completion"
)
def send_training_notification(
    status: str,
    regression_best: str,
    classification_best: str,
    saved_paths: Dict[str, str]
) -> bool:
    """
    Task: Send notification about training results.
    """
    logger = get_run_logger()
    
    message = f"""
Training Pipeline {status.upper()}

Best Models:
- Regression: {regression_best}
- Classification: {classification_best}

Saved to:
{json.dumps(saved_paths, indent=2)}
    """
    
    logger.info(f"NOTIFICATION: {message}")
    
    return True


@flow(
    name="Training Pipeline",
    description="Automated model training and evaluation pipeline",
    version="1.0",
    retries=1
)
def training_pipeline(
    use_synthetic_data: bool = True,
    n_samples: int = 1000,
    forecast_horizon: int = 1
) -> Dict[str, Any]:
    """
    Training Pipeline Flow
    
    Main orchestration flow for model training.
    
    Args:
        use_synthetic_data: Use synthetic data for testing
        n_samples: Number of samples if using synthetic
        forecast_horizon: Hours ahead to predict
    
    Returns:
        Pipeline execution summary
    """
    logger = get_run_logger()
    
    logger.info("Starting Training Pipeline")
    start_time = datetime.now()
    
    results = {
        "status": "success",
        "execution_time": None,
        "errors": []
    }
    
    try:
        # Step 1: Load data
        logger.info("Step 1: Loading training data")
        df = load_training_data(use_synthetic=use_synthetic_data, n_samples=n_samples)
        
        # Step 2: Prepare features
        logger.info("Step 2: Preparing features")
        data = prepare_features(df, forecast_horizon=forecast_horizon)
        
        # Step 3: Train regression models
        logger.info("Step 3: Training regression models")
        regression_results = train_regression(data)
        
        # Step 4: Train classification models
        logger.info("Step 4: Training classification models")
        classification_results = train_classification(data)
        
        # Step 5: Train time series models
        logger.info("Step 5: Training time series models")
        timeseries_results = train_timeseries(data)
        
        # Step 6: Save models
        logger.info("Step 6: Saving models")
        saved_paths = save_models(
            regression_results,
            classification_results,
            timeseries_results
        )
        
        # Step 7: Generate report
        logger.info("Step 7: Generating report")
        report = generate_report(
            regression_results,
            classification_results,
            timeseries_results
        )
        
        # Step 8: Send notification
        send_training_notification(
            "success",
            regression_results.get("best_model", "N/A"),
            classification_results.get("best_model", "N/A"),
            saved_paths
        )
        
        results["regression_best"] = regression_results.get("best_model")
        results["classification_best"] = classification_results.get("best_model")
        results["saved_paths"] = saved_paths
        results["report"] = report
        
    except Exception as e:
        results["status"] = "failed"
        results["errors"].append(str(e))
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
    
    results["execution_time"] = str(datetime.now() - start_time)
    
    logger.info(f"Training Pipeline completed with status: {results['status']}")
    
    return results


if __name__ == "__main__":
    # Run pipeline manually
    print("Running Training Pipeline")
    print("=" * 50)
    
    results = training_pipeline(
        use_synthetic_data=True,
        n_samples=500
    )
    
    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Time: {results['execution_time']}")
    print(f"  Best Regression: {results.get('regression_best')}")
    print(f"  Best Classification: {results.get('classification_best')}")
    
    if results.get("report"):
        print("\n" + results["report"])
