"""
Time Series Models for AQI Forecasting
======================================

This module implements time series models for AQI forecasting.

Models Implemented:
1. ARIMA - AutoRegressive Integrated Moving Average
2. Prophet - Facebook's forecasting library
3. LSTM - Long Short-Term Memory neural network

Why Time Series Models?
- AQI data is inherently temporal
- Past patterns help predict future values
- Captures trends, seasonality, and cyclical patterns
- Designed specifically for sequential data

Forecast Horizon:
- Default: 3 days (72 hours) ahead
- Can be adjusted based on requirements
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import joblib
import warnings

warnings.filterwarnings("ignore")

# Time series models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class TimeSeriesModels:
    """
    Time series models for AQI forecasting.
    
    This class provides:
    - ARIMA for traditional time series analysis
    - Prophet for automatic seasonality detection
    - LSTM for deep learning-based forecasting
    
    Usage:
        >>> models = TimeSeriesModels()
        >>> models.train_prophet(df, "aqi")
        >>> forecast = models.forecast_prophet(periods=72)  # 3 days
    """
    
    def __init__(self, forecast_horizon: int = 72):
        """
        Initialize time series models.
        
        Args:
            forecast_horizon: Number of periods to forecast (hours)
        """
        self.settings = get_settings()
        self.models_dir = Path(self.settings.app.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.forecast_horizon = forecast_horizon
        
        # Model storage
        self.arima_model = None
        self.prophet_model = None
        self.lstm_model = None
        
        # Scalers for LSTM
        self.lstm_scaler = MinMaxScaler()
        
        # Store training data info
        self.training_info = {}
        
        logger.info(f"Initialized TimeSeriesModels with horizon={forecast_horizon}")
        logger.info(f"Available: ARIMA={HAS_STATSMODELS}, Prophet={HAS_PROPHET}, LSTM={HAS_TENSORFLOW}")
    
    # ========================
    # ARIMA MODEL
    # ========================
    
    @log_execution_time
    def train_arima(
        self,
        series: pd.Series,
        order: Tuple[int, int, int] = (5, 1, 0)
    ) -> Dict[str, Any]:
        """
        Train ARIMA model on time series data.
        
        ARIMA (AutoRegressive Integrated Moving Average):
        - AR (p): Uses past values to predict future
        - I (d): Differencing to make series stationary
        - MA (q): Uses past forecast errors
        
        Args:
            series: Time series data (indexed by datetime)
            order: (p, d, q) parameters
        
        Returns:
            Training results and diagnostics
        """
        if not HAS_STATSMODELS:
            logger.error("statsmodels not installed")
            return {"error": "statsmodels not installed"}
        
        logger.info(f"Training ARIMA{order} on {len(series)} observations")
        
        # Ensure series is properly indexed
        if not isinstance(series.index, pd.DatetimeIndex):
            if "timestamp" in series.index.names or hasattr(series.index, 'to_datetime'):
                series.index = pd.to_datetime(series.index)
        
        # Fill missing values
        series = series.fillna(method="ffill").fillna(method="bfill")
        
        # Check stationarity
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < 0.05
        logger.info(f"ADF test p-value: {adf_result[1]:.4f}, Stationary: {is_stationary}")
        
        try:
            # Fit ARIMA model
            self.arima_model = ARIMA(series, order=order)
            self.arima_fit = self.arima_model.fit()
            
            # Get in-sample predictions
            predictions = self.arima_fit.fittedvalues
            
            # Calculate metrics
            mse = mean_squared_error(series[1:], predictions[1:])  # Skip first due to differencing
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(series[1:], predictions[1:])
            
            results = {
                "order": order,
                "aic": self.arima_fit.aic,
                "bic": self.arima_fit.bic,
                "rmse": rmse,
                "mae": mae,
                "is_stationary": is_stationary,
                "n_observations": len(series)
            }
            
            self.training_info["arima"] = results
            
            logger.info(f"ARIMA trained: RMSE={rmse:.2f}, MAE={mae:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return {"error": str(e)}
    
    def forecast_arima(self, periods: int = None) -> pd.DataFrame:
        """
        Generate forecast using trained ARIMA model.
        
        Args:
            periods: Number of periods to forecast
        
        Returns:
            DataFrame with forecast and confidence intervals
        """
        if self.arima_fit is None:
            raise ValueError("ARIMA model not trained")
        
        periods = periods or self.forecast_horizon
        
        # Generate forecast
        forecast = self.arima_fit.forecast(steps=periods)
        conf_int = self.arima_fit.get_forecast(steps=periods).conf_int()
        
        # Create DataFrame
        last_date = self.arima_fit.data.dates[-1]
        if isinstance(last_date, (int, np.integer)):
            # If index is not datetime, create simple index
            forecast_index = range(len(forecast))
        else:
            forecast_index = pd.date_range(
                start=last_date + timedelta(hours=1),
                periods=periods,
                freq="H"
            )
        
        result = pd.DataFrame({
            "forecast": forecast.values if hasattr(forecast, 'values') else forecast,
            "lower_ci": conf_int.iloc[:, 0].values if hasattr(conf_int, 'iloc') else conf_int[:, 0],
            "upper_ci": conf_int.iloc[:, 1].values if hasattr(conf_int, 'iloc') else conf_int[:, 1]
        }, index=forecast_index)
        
        return result
    
    # ========================
    # PROPHET MODEL
    # ========================
    
    @log_execution_time
    def train_prophet(
        self,
        df: pd.DataFrame,
        target_column: str = "aqi",
        timestamp_column: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Train Prophet model on time series data.
        
        Prophet automatically handles:
        - Yearly, weekly, and daily seasonality
        - Holiday effects
        - Trend changepoints
        - Missing data
        
        Args:
            df: DataFrame with timestamp and target columns
            target_column: Column to forecast
            timestamp_column: Column with timestamps
        
        Returns:
            Training results
        """
        if not HAS_PROPHET:
            logger.error("prophet not installed. Run: pip install prophet")
            return {"error": "prophet not installed"}
        
        logger.info(f"Training Prophet on {len(df)} observations")
        
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = df[[timestamp_column, target_column]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df = prophet_df.dropna()
        
        try:
            # Initialize Prophet with sensible defaults for AQI
            self.prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05,  # Regularization
                seasonality_mode="additive"
            )
            
            # Fit model
            self.prophet_model.fit(prophet_df)
            
            # Make in-sample predictions for metrics
            forecast = self.prophet_model.predict(prophet_df)
            
            # Calculate metrics
            y_true = prophet_df["y"].values
            y_pred = forecast["yhat"].values
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            
            results = {
                "rmse": rmse,
                "mae": mae,
                "n_observations": len(prophet_df),
                "components": ["trend", "yearly", "weekly", "daily"]
            }
            
            self.training_info["prophet"] = results
            
            logger.info(f"Prophet trained: RMSE={rmse:.2f}, MAE={mae:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            return {"error": str(e)}
    
    def forecast_prophet(self, periods: int = None) -> pd.DataFrame:
        """
        Generate forecast using trained Prophet model.
        
        Args:
            periods: Number of hours to forecast
        
        Returns:
            DataFrame with forecast and components
        """
        if self.prophet_model is None:
            raise ValueError("Prophet model not trained")
        
        periods = periods or self.forecast_horizon
        
        # Create future dataframe
        future = self.prophet_model.make_future_dataframe(periods=periods, freq="H")
        
        # Generate forecast
        forecast = self.prophet_model.predict(future)
        
        # Get only future predictions
        result = forecast.tail(periods)[[
            "ds", "yhat", "yhat_lower", "yhat_upper", 
            "trend", "yearly", "weekly", "daily"
        ]].copy()
        
        result.columns = [
            "timestamp", "forecast", "lower_ci", "upper_ci",
            "trend", "yearly", "weekly", "daily"
        ]
        
        return result.set_index("timestamp")
    
    # ========================
    # LSTM MODEL
    # ========================
    
    @log_execution_time
    def train_lstm(
        self,
        df: pd.DataFrame,
        target_column: str = "aqi",
        feature_columns: List[str] = None,
        sequence_length: int = 24,
        epochs: int = 50,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Train LSTM neural network for time series forecasting.
        
        LSTM is particularly good at:
        - Learning long-term dependencies
        - Handling multiple input features
        - Capturing complex nonlinear patterns
        
        Args:
            df: DataFrame with features and target
            target_column: Column to forecast
            feature_columns: Columns to use as features (None = just target)
            sequence_length: Number of past steps to use as input
            epochs: Training epochs
            batch_size: Training batch size
        
        Returns:
            Training results
        """
        if not HAS_TENSORFLOW:
            logger.error("tensorflow not installed")
            return {"error": "tensorflow not installed"}
        
        logger.info(f"Training LSTM on {len(df)} observations")
        
        # Prepare features
        if feature_columns is None:
            feature_columns = [target_column]
        
        # Ensure all columns exist
        feature_columns = [c for c in feature_columns if c in df.columns]
        if target_column not in feature_columns:
            feature_columns.append(target_column)
        
        # Get data
        data = df[feature_columns].copy()
        data = data.fillna(method="ffill").fillna(method="bfill")
        
        # Scale data
        scaled_data = self.lstm_scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, sequence_length)
        
        # Split into train/validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        try:
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, len(feature_columns))),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dense(1)
            ])
            
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            
            # Early stopping
            early_stop = EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            self.lstm_model = model
            self.lstm_sequence_length = sequence_length
            self.lstm_feature_columns = feature_columns
            self.lstm_target_index = feature_columns.index(target_column)
            
            # Evaluate
            val_pred = model.predict(X_val, verbose=0)
            
            # Inverse transform predictions
            val_pred_full = np.zeros((len(val_pred), len(feature_columns)))
            val_pred_full[:, self.lstm_target_index] = val_pred.flatten()
            val_pred_unscaled = self.lstm_scaler.inverse_transform(val_pred_full)[:, self.lstm_target_index]
            
            y_val_full = np.zeros((len(y_val), len(feature_columns)))
            y_val_full[:, self.lstm_target_index] = y_val.flatten()
            y_val_unscaled = self.lstm_scaler.inverse_transform(y_val_full)[:, self.lstm_target_index]
            
            rmse = np.sqrt(mean_squared_error(y_val_unscaled, val_pred_unscaled))
            mae = mean_absolute_error(y_val_unscaled, val_pred_unscaled)
            
            results = {
                "rmse": float(rmse),
                "mae": float(mae),
                "n_observations": len(df),
                "sequence_length": sequence_length,
                "n_features": len(feature_columns),
                "epochs_trained": len(history.history["loss"]),
                "final_loss": float(history.history["loss"][-1]),
                "final_val_loss": float(history.history["val_loss"][-1])
            }
            
            self.training_info["lstm"] = results
            
            logger.info(f"LSTM trained: RMSE={rmse:.2f}, MAE={mae:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return {"error": str(e)}
    
    def _create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, -1])  # Last column is target
        
        return np.array(X), np.array(y)
    
    def forecast_lstm(
        self,
        recent_data: pd.DataFrame,
        periods: int = None
    ) -> pd.DataFrame:
        """
        Generate forecast using trained LSTM model.
        
        Args:
            recent_data: Recent data to use for initial sequence
            periods: Number of steps to forecast
        
        Returns:
            DataFrame with forecasts
        """
        if self.lstm_model is None:
            raise ValueError("LSTM model not trained")
        
        periods = periods or self.forecast_horizon
        
        # Prepare recent data
        data = recent_data[self.lstm_feature_columns].copy()
        data = data.fillna(method="ffill").fillna(method="bfill")
        
        # Get last sequence
        if len(data) < self.lstm_sequence_length:
            raise ValueError(f"Need at least {self.lstm_sequence_length} observations")
        
        scaled_data = self.lstm_scaler.transform(data)
        sequence = scaled_data[-self.lstm_sequence_length:]
        
        # Generate forecasts iteratively
        forecasts = []
        current_seq = sequence.copy()
        
        for _ in range(periods):
            # Predict next step
            pred = self.lstm_model.predict(
                current_seq.reshape(1, self.lstm_sequence_length, -1),
                verbose=0
            )[0, 0]
            
            # Store prediction
            forecasts.append(pred)
            
            # Update sequence (roll and add prediction)
            new_row = current_seq[-1].copy()
            new_row[self.lstm_target_index] = pred
            current_seq = np.vstack([current_seq[1:], new_row])
        
        # Inverse transform
        forecasts_full = np.zeros((len(forecasts), len(self.lstm_feature_columns)))
        forecasts_full[:, self.lstm_target_index] = forecasts
        forecasts_unscaled = self.lstm_scaler.inverse_transform(forecasts_full)[:, self.lstm_target_index]
        
        # Create forecast dataframe
        last_timestamp = pd.to_datetime(recent_data.index[-1]) if isinstance(recent_data.index[-1], str) else recent_data.index[-1]
        forecast_index = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=periods,
            freq="H"
        )
        
        result = pd.DataFrame({
            "forecast": forecasts_unscaled
        }, index=forecast_index)
        
        return result
    
    # ========================
    # UTILITY METHODS
    # ========================
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        results = []
        
        for model_name, info in self.training_info.items():
            if "error" not in info:
                results.append({
                    "model": model_name.upper(),
                    "rmse": info.get("rmse", float("nan")),
                    "mae": info.get("mae", float("nan")),
                    "observations": info.get("n_observations", 0)
                })
        
        if not results:
            return pd.DataFrame()
        
        return pd.DataFrame(results).sort_values("rmse")
    
    def save(self, filename: str = None) -> Path:
        """Save all trained models."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"timeseries_models_{timestamp}"
        
        filepath = self.models_dir / f"{filename}.joblib"
        
        save_data = {
            "arima_fit": self.arima_fit if hasattr(self, "arima_fit") else None,
            "prophet_model": self.prophet_model,
            "lstm_scaler": self.lstm_scaler,
            "lstm_sequence_length": getattr(self, "lstm_sequence_length", None),
            "lstm_feature_columns": getattr(self, "lstm_feature_columns", None),
            "lstm_target_index": getattr(self, "lstm_target_index", None),
            "training_info": self.training_info,
            "forecast_horizon": self.forecast_horizon
        }
        
        # Save LSTM model separately if exists
        if self.lstm_model is not None:
            lstm_path = self.models_dir / f"{filename}_lstm.keras"
            self.lstm_model.save(lstm_path)
            save_data["lstm_path"] = str(lstm_path)
        
        joblib.dump(save_data, filepath)
        logger.info(f"Saved time series models to {filepath}")
        
        return filepath


def train_timeseries_models(
    df: pd.DataFrame,
    target_column: str = "aqi"
) -> Tuple[TimeSeriesModels, Dict[str, Any]]:
    """
    Convenience function to train all time series models.
    
    Args:
        df: DataFrame with timestamp and target
        target_column: Column to forecast
    
    Returns:
        Tuple of (models, results)
    """
    models = TimeSeriesModels()
    results = {}
    
    # Train Prophet if available
    if HAS_PROPHET:
        results["prophet"] = models.train_prophet(df, target_column)
    
    # Train ARIMA if available
    if HAS_STATSMODELS and "timestamp" in df.columns:
        series = df.set_index("timestamp")[target_column]
        results["arima"] = models.train_arima(series)
    
    # Train LSTM if available
    if HAS_TENSORFLOW:
        results["lstm"] = models.train_lstm(df, target_column)
    
    return models, results


if __name__ == "__main__":
    # Test time series models
    print("Testing Time Series Models")
    print("=" * 50)
    
    # Generate sample data
    from src.data.feature_engineering import generate_synthetic_training_data
    
    print(f"\nAvailable models:")
    print(f"  ARIMA (statsmodels): {HAS_STATSMODELS}")
    print(f"  Prophet: {HAS_PROPHET}")
    print(f"  LSTM (TensorFlow): {HAS_TENSORFLOW}")
    
    print("\n1. Generating training data:")
    df = generate_synthetic_training_data(n_samples=500)
    print(f"   Generated {len(df)} samples")
    
    # Initialize models
    models = TimeSeriesModels(forecast_horizon=72)  # 3 days
    
    # Train ARIMA
    if HAS_STATSMODELS:
        print("\n2a. Training ARIMA:")
        series = df.set_index("timestamp")["aqi"]
        arima_results = models.train_arima(series)
        if "error" not in arima_results:
            print(f"   RMSE: {arima_results['rmse']:.2f}")
    
    # Train Prophet
    if HAS_PROPHET:
        print("\n2b. Training Prophet:")
        prophet_results = models.train_prophet(df, "aqi")
        if "error" not in prophet_results:
            print(f"   RMSE: {prophet_results['rmse']:.2f}")
    
    # Train LSTM
    if HAS_TENSORFLOW:
        print("\n2c. Training LSTM:")
        lstm_results = models.train_lstm(df, "aqi", epochs=10)
        if "error" not in lstm_results:
            print(f"   RMSE: {lstm_results['rmse']:.2f}")
    
    # Generate forecasts
    print("\n3. Generating 72-hour (3-day) forecasts:")
    
    if HAS_PROPHET and models.prophet_model:
        forecast = models.forecast_prophet(72)
        print(f"   Prophet forecast range: {forecast['forecast'].min():.1f} - {forecast['forecast'].max():.1f}")
    
    # Compare models
    print("\n4. Model Comparison:")
    comparison = models.compare_models()
    if len(comparison) > 0:
        print(comparison.to_string(index=False))
    
    # Save models
    print("\n5. Saving models:")
    filepath = models.save("test_timeseries_models")
    print(f"   Saved to: {filepath}")
    
    print("\n" + "=" * 50)
    print("Time Series Models Test Complete!")
