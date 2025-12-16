"""
Regression Models for AQI Prediction
====================================

This module implements regression models for predicting continuous AQI values.

Models Implemented:
1. Linear Regression - Simple baseline model
2. Ridge Regression - L2 regularization
3. Lasso Regression - L1 regularization (feature selection)
4. Random Forest - Ensemble of decision trees
5. XGBoost - Gradient boosting

Why Multiple Models?
- Different models have different strengths
- Comparison helps identify best approach
- Ensemble methods often outperform single models
- Understanding trade-offs between interpretability and performance

Model Selection Process:
1. Train all models
2. Evaluate on validation set
3. Compare metrics (RMSE, MAE, R²)
4. Select best model for deployment
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import joblib

# Scikit-learn imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class RegressionModels:
    """
    Container for regression models with unified training and prediction interface.
    
    This class:
    - Manages multiple regression models
    - Handles preprocessing (scaling)
    - Provides consistent training/prediction API
    - Supports model persistence (save/load)
    
    Usage:
        >>> models = RegressionModels()
        >>> models.train(X_train, y_train)
        >>> predictions = models.predict(X_test, model_name="xgboost")
        >>> metrics = models.evaluate(X_test, y_test)
    """
    
    def __init__(self, models_to_use: List[str] = None):
        """
        Initialize regression models.
        
        Args:
            models_to_use: List of model names to use. Options:
                - "linear", "ridge", "lasso", "elastic_net"
                - "random_forest", "gradient_boosting"
                - "xgboost"
                If None, uses all available models.
        """
        self.settings = get_settings()
        self.models_dir = Path(self.settings.app.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        all_models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=1.0),
            "elastic_net": ElasticNet(alpha=1.0, l1_ratio=0.5),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            all_models["xgboost"] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        # Filter to requested models
        if models_to_use:
            self.models = {k: v for k, v in all_models.items() if k in models_to_use}
        else:
            self.models = all_models
        
        # Scaler for feature normalization
        self.scaler = StandardScaler()
        
        # Track trained status
        self.is_trained = {name: False for name in self.models.keys()}
        
        # Store training metadata
        self.training_metadata = {}
        
        logger.info(f"Initialized {len(self.models)} regression models: {list(self.models.keys())}")
    
    @log_execution_time
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        scale_features: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all regression models.
        
        Args:
            X: Feature DataFrame
            y: Target Series (AQI values)
            validation_split: Fraction for validation set
            scale_features: Whether to scale features
        
        Returns:
            Dictionary of training results for each model
        
        Example:
            >>> X, y = feature_engineer.prepare_training_data(df)
            >>> results = models.train(X, y)
            >>> print(results["xgboost"]["rmse"])
        """
        logger.info(f"Training {len(self.models)} models on {len(X)} samples")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features if requested
        if scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y,
            test_size=validation_split,
            random_state=42
        )
        
        logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                self.is_trained[name] = True
                
                # Make predictions
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Calculate metrics
                train_metrics = self._calculate_metrics(y_train, y_train_pred)
                val_metrics = self._calculate_metrics(y_val, y_val_pred)
                
                results[name] = {
                    "train_rmse": train_metrics["rmse"],
                    "train_mae": train_metrics["mae"],
                    "train_r2": train_metrics["r2"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                    "val_r2": val_metrics["r2"],
                }
                
                logger.info(
                    f"  {name}: Val RMSE={val_metrics['rmse']:.2f}, "
                    f"R²={val_metrics['r2']:.3f}"
                )
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {"error": str(e)}
        
        # Store training metadata
        self.training_metadata = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_features": len(X.columns),
            "feature_names": list(X.columns),
            "results": results
        }
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
    
    @log_execution_time
    def predict(
        self,
        X: pd.DataFrame,
        model_name: str = None,
        scale_features: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions using trained models.
        
        Args:
            X: Feature DataFrame
            model_name: Specific model to use (None = all models)
            scale_features: Whether to scale features
        
        Returns:
            Predictions (array if single model, dict if all models)
        """
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        if scale_features:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X
        
        if model_name:
            # Single model prediction
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            if not self.is_trained[model_name]:
                raise ValueError(f"Model {model_name} not trained")
            
            return self.models[model_name].predict(X_scaled)
        else:
            # All models prediction
            predictions = {}
            for name, model in self.models.items():
                if self.is_trained[name]:
                    predictions[name] = model.predict(X_scaled)
            return predictions
    
    @log_execution_time
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_features: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test data.
        
        Args:
            X: Feature DataFrame
            y: True target values
            scale_features: Whether to scale features
        
        Returns:
            Dictionary of metrics for each model
        """
        results = {}
        predictions = self.predict(X, scale_features=scale_features)
        
        for name, y_pred in predictions.items():
            results[name] = self._calculate_metrics(y.values, y_pred)
            
        return results
    
    def get_best_model(self, metric: str = "rmse") -> Tuple[str, Any]:
        """
        Get the best performing model based on validation metrics.
        
        Args:
            metric: Metric to use ("rmse", "mae", "r2")
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if not self.training_metadata.get("results"):
            raise ValueError("Models not trained yet")
        
        results = self.training_metadata["results"]
        
        # Filter out models with errors
        valid_results = {
            k: v for k, v in results.items() 
            if "error" not in v and f"val_{metric}" in v
        }
        
        if not valid_results:
            raise ValueError("No valid results available")
        
        # Find best model
        if metric == "r2":
            # Higher R² is better
            best_name = max(valid_results, key=lambda k: valid_results[k][f"val_{metric}"])
        else:
            # Lower RMSE/MAE is better
            best_name = min(valid_results, key=lambda k: valid_results[k][f"val_{metric}"])
        
        logger.info(
            f"Best model ({metric}): {best_name} "
            f"with val_{metric}={valid_results[best_name][f'val_{metric}']:.3f}"
        )
        
        return best_name, self.models[best_name]
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            model_name: Model to get importance from
                        (random_forest, gradient_boosting, xgboost)
        
        Returns:
            DataFrame with feature importances
        """
        if model_name is None:
            # Use best model
            model_name = self.get_best_model()[0]
        
        model = self.models.get(model_name)
        
        if model is None or not self.is_trained.get(model_name):
            raise ValueError(f"Model {model_name} not available or not trained")
        
        if not hasattr(model, "feature_importances_"):
            raise ValueError(f"Model {model_name} doesn't support feature importance")
        
        feature_names = self.training_metadata.get("feature_names", [])
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    @log_execution_time
    def hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        param_grid: Dict = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            model_name: Model to tune
            param_grid: Parameters to search
            cv: Number of cross-validation folds
        
        Returns:
            Best parameters and scores
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Default parameter grids
        default_grids = {
            "ridge": {"alpha": [0.1, 1.0, 10.0, 100.0]},
            "lasso": {"alpha": [0.01, 0.1, 1.0, 10.0]},
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5, 10]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.3]
            }
        }
        
        if param_grid is None:
            param_grid = default_grids.get(model_name, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid for {model_name}")
            return {}
        
        logger.info(f"Tuning {model_name} with {cv}-fold CV...")
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        results = {
            "best_params": grid_search.best_params_,
            "best_score": -grid_search.best_score_,  # Negate to get RMSE
            "cv_results": pd.DataFrame(grid_search.cv_results_)
        }
        
        logger.info(f"Best params: {results['best_params']}, RMSE: {results['best_score']:.2f}")
        
        return results
    
    def save(self, filename: str = None) -> Path:
        """
        Save all trained models to disk.
        
        Args:
            filename: Output filename (without extension)
        
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regression_models_{timestamp}"
        
        filepath = self.models_dir / f"{filename}.joblib"
        
        save_data = {
            "models": {k: v for k, v in self.models.items() if self.is_trained[k]},
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "training_metadata": self.training_metadata
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Saved models to {filepath}")
        
        return filepath
    
    def load(self, filepath: Union[str, Path]) -> None:
        """
        Load models from disk.
        
        Args:
            filepath: Path to saved models file
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        self.models.update(save_data["models"])
        self.scaler = save_data["scaler"]
        self.is_trained = save_data["is_trained"]
        self.training_metadata = save_data["training_metadata"]
        
        logger.info(f"Loaded models from {filepath}")


def train_regression_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_to_use: List[str] = None
) -> Tuple[RegressionModels, Dict[str, Dict]]:
    """
    Convenience function to train regression models.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        models_to_use: List of model names
    
    Returns:
        Tuple of (trained models, results)
    
    Example:
        >>> models, results = train_regression_models(X, y)
        >>> print(results)
    """
    regression = RegressionModels(models_to_use)
    results = regression.train(X, y)
    return regression, results


if __name__ == "__main__":
    # Test regression models
    print("Testing Regression Models")
    print("=" * 50)
    
    # Generate sample data
    from src.data.feature_engineering import generate_synthetic_training_data, FeatureEngineer
    
    print("\n1. Generating training data:")
    df = generate_synthetic_training_data(n_samples=1000)
    
    fe = FeatureEngineer()
    X, y = fe.prepare_training_data(df, target_column="aqi", forecast_horizon=1)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    
    # Train models
    print("\n2. Training regression models:")
    models = RegressionModels()
    results = models.train(X, y)
    
    # Print results
    print("\n3. Training Results:")
    print("-" * 60)
    for name, metrics in results.items():
        if "error" not in metrics:
            print(f"   {name:20s}: RMSE={metrics['val_rmse']:7.2f}, R²={metrics['val_r2']:.3f}")
    
    # Get best model
    print("\n4. Best Model:")
    best_name, best_model = models.get_best_model(metric="rmse")
    print(f"   {best_name}")
    
    # Feature importance
    print("\n5. Top 10 Feature Importance:")
    try:
        importance = models.get_feature_importance()
        print(importance.head(10).to_string(index=False))
    except Exception as e:
        print(f"   Error: {e}")
    
    # Save models
    print("\n6. Saving models:")
    filepath = models.save("test_regression_models")
    print(f"   Saved to: {filepath}")
    
    print("\n" + "=" * 50)
    print("Regression Models Test Complete!")
