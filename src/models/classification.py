"""
Classification Models for AQI Category Prediction
==================================================

This module implements classification models for predicting AQI categories.

AQI Categories (US EPA Standard):
- Good (0-50)
- Moderate (51-100)
- Unhealthy for Sensitive Groups (101-150)
- Unhealthy (151-200)
- Very Unhealthy (201-300)
- Hazardous (301-500)

Models Implemented:
1. Logistic Regression - Baseline classifier
2. Random Forest Classifier - Ensemble method
3. XGBoost Classifier - Gradient boosting
4. Multi-layer Perceptron - Neural network

Why Classification?
- Sometimes we care more about category than exact value
- "Will tomorrow be unhealthy?" is actionable
- Health advisories are category-based
- Classification can be more robust to outliers
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import joblib

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config.settings import get_settings, AQI_CATEGORIES
from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class ClassificationModels:
    """
    Classification models for AQI category prediction.
    
    This class predicts which AQI category (Good, Moderate, Unhealthy, etc.)
    a future observation will fall into.
    
    Usage:
        >>> models = ClassificationModels()
        >>> models.train(X_train, y_train)
        >>> predictions = models.predict(X_test)
        >>> metrics = models.evaluate(X_test, y_test)
    """
    
    # AQI category labels (ordered)
    CATEGORY_LABELS = [
        "Good",
        "Moderate", 
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ]
    
    def __init__(self, models_to_use: List[str] = None):
        """
        Initialize classification models.
        
        Args:
            models_to_use: List of model names. Options:
                - "logistic", "random_forest", "mlp", "xgboost"
        """
        self.settings = get_settings()
        self.models_dir = Path(self.settings.app.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        all_models = {
            "logistic": LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                random_state=42
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                max_iter=500,
                random_state=42
            ),
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            all_models["xgboost"] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric="mlogloss"
            )
        
        # Filter to requested models
        if models_to_use:
            self.models = {k: v for k, v in all_models.items() if k in models_to_use}
        else:
            self.models = all_models
        
        # Scaler and encoder
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Track trained status
        self.is_trained = {name: False for name in self.models.keys()}
        
        # Store training metadata
        self.training_metadata = {}
        
        logger.info(f"Initialized {len(self.models)} classification models")
    
    @log_execution_time
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        scale_features: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all classification models.
        
        Args:
            X: Feature DataFrame
            y: Target Series (AQI categories as strings or encoded integers)
            validation_split: Fraction for validation set
            scale_features: Whether to scale features
        
        Returns:
            Dictionary of training results for each model
        """
        logger.info(f"Training {len(self.models)} classifiers on {len(X)} samples")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode labels if string
        if y.dtype == object or isinstance(y.iloc[0], str):
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            # Assume already encoded
            y_encoded = y.values
            # Fit encoder for inverse transform later
            self.label_encoder.fit(self.CATEGORY_LABELS[:len(np.unique(y_encoded))])
        
        # Scale features
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
            X_scaled, y_encoded,
            test_size=validation_split,
            random_state=42,
            stratify=y_encoded
        )
        
        logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Class distribution: {dict(zip(unique, counts))}")
        
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
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1": train_metrics["f1_weighted"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1_weighted"],
                    "val_precision": val_metrics["precision_weighted"],
                    "val_recall": val_metrics["recall_weighted"],
                }
                
                logger.info(
                    f"  {name}: Val Accuracy={val_metrics['accuracy']:.3f}, "
                    f"F1={val_metrics['f1_weighted']:.3f}"
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
            "class_labels": list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else [],
            "results": results
        }
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate classification metrics."""
        accuracy = accuracy_score(y_true, y_pred)
        
        # Average methods for multiclass
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        
        # Also calculate macro (unweighted) metrics
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        
        return {
            "accuracy": float(accuracy),
            "precision_weighted": float(precision),
            "recall_weighted": float(recall),
            "f1_weighted": float(f1),
            "f1_macro": float(f1_macro)
        }
    
    @log_execution_time
    def predict(
        self,
        X: pd.DataFrame,
        model_name: str = None,
        scale_features: bool = True,
        return_proba: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make predictions using trained models.
        
        Args:
            X: Feature DataFrame
            model_name: Specific model to use
            scale_features: Whether to scale features
            return_proba: Whether to return probabilities instead of classes
        
        Returns:
            Predictions (class labels or probabilities)
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
            
            model = self.models[model_name]
            
            if return_proba and hasattr(model, "predict_proba"):
                return model.predict_proba(X_scaled)
            else:
                return model.predict(X_scaled)
        else:
            # All models prediction
            predictions = {}
            for name, model in self.models.items():
                if self.is_trained[name]:
                    if return_proba and hasattr(model, "predict_proba"):
                        predictions[name] = model.predict_proba(X_scaled)
                    else:
                        predictions[name] = model.predict(X_scaled)
            return predictions
    
    def predict_categories(
        self,
        X: pd.DataFrame,
        model_name: str = None
    ) -> List[str]:
        """
        Predict AQI categories as human-readable strings.
        
        Args:
            X: Feature DataFrame
            model_name: Model to use
        
        Returns:
            List of category names
        """
        predictions = self.predict(X, model_name)
        
        if isinstance(predictions, dict):
            # Return predictions from best model
            model_name = self.get_best_model()[0]
            predictions = predictions[model_name]
        
        # Decode labels
        if hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.inverse_transform(predictions).tolist()
        
        return predictions.tolist()
    
    @log_execution_time
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_features: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models on test data."""
        results = {}
        
        # Encode labels if needed
        if y.dtype == object or isinstance(y.iloc[0], str):
            y_encoded = self.label_encoder.transform(y)
        else:
            y_encoded = y.values
        
        predictions = self.predict(X, scale_features=scale_features)
        
        for name, y_pred in predictions.items():
            results[name] = self._calculate_metrics(y_encoded, y_pred)
        
        return results
    
    def get_classification_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = None
    ) -> str:
        """
        Get detailed classification report.
        
        Returns formatted classification report with per-class metrics.
        """
        if model_name is None:
            model_name = self.get_best_model()[0]
        
        # Get predictions
        y_pred = self.predict(X, model_name)
        
        # Encode labels if needed
        if y.dtype == object or isinstance(y.iloc[0], str):
            y_true = self.label_encoder.transform(y)
        else:
            y_true = y.values
        
        # Get class names
        if hasattr(self.label_encoder, 'classes_'):
            target_names = self.label_encoder.classes_
        else:
            target_names = None
        
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def get_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str = None
    ) -> pd.DataFrame:
        """Get confusion matrix as DataFrame."""
        if model_name is None:
            model_name = self.get_best_model()[0]
        
        y_pred = self.predict(X, model_name)
        
        if y.dtype == object or isinstance(y.iloc[0], str):
            y_true = self.label_encoder.transform(y)
        else:
            y_true = y.values
        
        cm = confusion_matrix(y_true, y_pred)
        
        if hasattr(self.label_encoder, 'classes_'):
            labels = self.label_encoder.classes_
        else:
            labels = [f"Class_{i}" for i in range(len(cm))]
        
        return pd.DataFrame(cm, index=labels, columns=labels)
    
    def get_best_model(self, metric: str = "f1") -> Tuple[str, Any]:
        """Get the best performing model."""
        if not self.training_metadata.get("results"):
            raise ValueError("Models not trained yet")
        
        results = self.training_metadata["results"]
        
        valid_results = {
            k: v for k, v in results.items()
            if "error" not in v and f"val_{metric}_weighted" in v
        }
        
        if not valid_results:
            # Fall back to accuracy
            valid_results = {
                k: v for k, v in results.items()
                if "error" not in v and "val_accuracy" in v
            }
            metric = "accuracy"
        
        if not valid_results:
            raise ValueError("No valid results available")
        
        # Higher is better for classification metrics
        metric_key = f"val_{metric}_weighted" if f"val_{metric}_weighted" in list(valid_results.values())[0] else f"val_{metric}"
        best_name = max(valid_results, key=lambda k: valid_results[k][metric_key])
        
        logger.info(f"Best classifier ({metric}): {best_name}")
        
        return best_name, self.models[best_name]
    
    def save(self, filename: str = None) -> Path:
        """Save all trained models to disk."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"classification_models_{timestamp}"
        
        filepath = self.models_dir / f"{filename}.joblib"
        
        save_data = {
            "models": {k: v for k, v in self.models.items() if self.is_trained[k]},
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "is_trained": self.is_trained,
            "training_metadata": self.training_metadata
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Saved classifiers to {filepath}")
        
        return filepath
    
    def load(self, filepath: Union[str, Path]) -> None:
        """Load models from disk."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        self.models.update(save_data["models"])
        self.scaler = save_data["scaler"]
        self.label_encoder = save_data["label_encoder"]
        self.is_trained = save_data["is_trained"]
        self.training_metadata = save_data["training_metadata"]
        
        logger.info(f"Loaded classifiers from {filepath}")


def train_classification_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_to_use: List[str] = None
) -> Tuple[ClassificationModels, Dict[str, Dict]]:
    """
    Convenience function to train classification models.
    
    Args:
        X: Feature DataFrame
        y: Target Series (category labels or encoded)
        models_to_use: List of model names
    
    Returns:
        Tuple of (trained models, results)
    """
    classifiers = ClassificationModels(models_to_use)
    results = classifiers.train(X, y)
    return classifiers, results


if __name__ == "__main__":
    # Test classification models
    print("Testing Classification Models")
    print("=" * 50)
    
    # Generate sample data
    from src.data.feature_engineering import generate_synthetic_training_data, FeatureEngineer
    
    print("\n1. Generating training data:")
    df = generate_synthetic_training_data(n_samples=1000)
    
    fe = FeatureEngineer()
    # Prepare features (no lag shift for simplicity)
    df = fe.add_lag_features(df)
    df = fe.add_rolling_features(df)
    df = fe.add_derived_features(df)
    df = df.dropna()
    
    # Get features and target (category)
    feature_cols = [c for c in df.columns if c not in [
        "timestamp", "city", "aqi_category_name", "aqi_category",
        "dominant_pollutant", "weather_condition"
    ]]
    X = df[feature_cols]
    y = df["aqi_category_name"]  # Use string labels
    
    print(f"   X shape: {X.shape}, y classes: {y.nunique()}")
    print(f"   Class distribution:\n{y.value_counts()}")
    
    # Train models
    print("\n2. Training classification models:")
    models = ClassificationModels()
    results = models.train(X, y)
    
    # Print results
    print("\n3. Training Results:")
    print("-" * 60)
    for name, metrics in results.items():
        if "error" not in metrics:
            print(f"   {name:15s}: Accuracy={metrics['val_accuracy']:.3f}, F1={metrics['val_f1']:.3f}")
    
    # Get best model
    print("\n4. Best Model:")
    best_name, best_model = models.get_best_model()
    print(f"   {best_name}")
    
    # Predict categories
    print("\n5. Sample Predictions:")
    sample = X.iloc[:5]
    predictions = models.predict_categories(sample, best_name)
    print(f"   {predictions}")
    
    # Save models
    print("\n6. Saving models:")
    filepath = models.save("test_classification_models")
    print(f"   Saved to: {filepath}")
    
    print("\n" + "=" * 50)
    print("Classification Models Test Complete!")
