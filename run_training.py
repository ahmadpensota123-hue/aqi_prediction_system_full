"""Quick test script to run training"""

import sys

sys.path.insert(0, ".")

from src.data.feature_engineering import (
    generate_synthetic_training_data,
    FeatureEngineer,
)
from src.models.regression import RegressionModels

print("=" * 60)
print("AQI PREDICTION SYSTEM - TRAINING DEMO")
print("=" * 60)

# Generate synthetic data
print("\n1. Generating synthetic training data...")
df = generate_synthetic_training_data(n_samples=300)
print(f"   Generated {len(df)} samples")

# Feature engineering
print("\n2. Engineering features...")
fe = FeatureEngineer()
df = fe.add_lag_features(df)
df = fe.add_rolling_features(df)
df = fe.add_derived_features(df)
df = df.dropna()
print(f"   After feature engineering: {len(df)} samples, {len(df.columns)} features")

# Prepare X and y
exclude = [
    "timestamp",
    "city",
    "aqi_category_name",
    "aqi_category",
    "dominant_pollutant",
    "weather_condition",
]
X = df[[c for c in df.columns if c not in exclude]]
y = df["aqi"]
print(f"   X shape: {X.shape}, y shape: {y.shape}")

# Train models
print("\n3. Training regression models...")
models = RegressionModels(models_to_use=["linear", "ridge", "random_forest", "xgboost"])
results = models.train(X, y)

# Print results
print("\n4. RESULTS:")
print("-" * 50)
for name, metrics in results.items():
    if "error" not in metrics:
        print(
            f"   {name:15s}: RMSE={metrics['val_rmse']:.2f}, R²={metrics['val_r2']:.3f}"
        )

# Best model
print("\n5. Best Model:")
best_name, best_model = models.get_best_model()
print(f"   {best_name}")

# Save
print("\n6. Saving models...")
path = models.save("demo_regression_models")
print(f"   Saved to: {path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
