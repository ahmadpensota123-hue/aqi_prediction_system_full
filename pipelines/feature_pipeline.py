"""
Feature Pipeline - Prefect Workflow
===================================

This pipeline handles automated data ingestion and feature engineering.

Pipeline Steps:
1. Fetch data from APIs (AQICN, OpenWeather)
2. Engineer features
3. Store features in Feature Store
4. Send notifications on completion/failure

Scheduling:
- Runs hourly to keep features up-to-date
- Can be triggered manually

Why Prefect?
- Modern workflow orchestration
- Easy retry handling
- Built-in scheduling
- Good observability (UI dashboard)
- Python-native API
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

# Import our modules
from src.config.settings import get_settings
from src.data.ingestion import DataIngestionService
from src.data.feature_engineering import FeatureEngineer
from src.feature_store.hopsworks_client import FeatureStoreManager


@task(
    name="Fetch AQI Data",
    description="Fetch data from AQICN and OpenWeather APIs",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(minutes=30),
)
def fetch_data(city: str) -> Dict[str, Any]:
    """
    Task: Fetch data from external APIs.

    Features:
    - 3 retries with 60-second delay
    - Caching to avoid duplicate API calls

    Args:
        city: City to fetch data for

    Returns:
        Raw data dictionary
    """
    logger = get_run_logger()
    logger.info(f"Fetching data for city: {city}")

    ingestion = DataIngestionService()
    data = ingestion.fetch_all_data(city)

    logger.info(f"Fetched data at {data.get('fetch_timestamp')}")

    return data


@task(name="Save Raw Data", description="Save raw data to disk for reproducibility")
def save_raw_data(data: Dict[str, Any], city: str) -> str:
    """
    Task: Save raw data to disk.

    Returns path to saved file.
    """
    logger = get_run_logger()

    ingestion = DataIngestionService()
    filepath = ingestion.save_raw_data(data, city)

    logger.info(f"Saved raw data to {filepath}")

    return str(filepath)


@task(name="Engineer Features", description="Create ML features from raw data")
def create_features(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Task: Create engineered features from raw data.

    Returns feature dictionary.
    """
    logger = get_run_logger()

    fe = FeatureEngineer()
    features = fe.create_features_from_raw(data)

    n_features = len([k for k in features.keys() if k not in ["city", "timestamp"]])
    logger.info(f"Created {n_features} features")

    return features


@task(name="Store Features", description="Store features in Feature Store")
def store_features(features: Dict[str, Any]) -> bool:
    """
    Task: Store features in feature store.

    Handles both cloud (Hopsworks) and local storage.
    """
    logger = get_run_logger()

    import pandas as pd

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Store in feature store
    fs_manager = FeatureStoreManager()
    fs_manager.initialize()
    success = fs_manager.ingest_features(df)
    fs_manager.close()

    logger.info(f"Features stored successfully: {success}")

    return success


@task(name="Send Notification", description="Send notification about pipeline status")
def send_notification(status: str, message: str, details: Dict = None) -> bool:
    """
    Task: Send notification (placeholder for email/Slack/Discord).

    In production, integrate with:
    - Email (via SMTP)
    - Slack (via webhook)
    - Discord (via webhook)
    - PagerDuty (for alerts)
    """
    logger = get_run_logger()

    # Log notification (placeholder for actual notification)
    logger.info(f"NOTIFICATION [{status}]: {message}")
    if details:
        logger.info(f"Details: {details}")

    # TODO: Implement actual notification
    # Example for Slack:
    # import requests
    # requests.post(SLACK_WEBHOOK, json={"text": message})

    return True


@flow(
    name="Feature Pipeline",
    description="Automated feature ingestion and engineering pipeline",
    version="1.0",
    retries=2,
    retry_delay_seconds=300,
)
def feature_pipeline(
    cities: List[str] = None, send_notifications: bool = True
) -> Dict[str, Any]:
    """
    Feature Pipeline Flow

    This is the main orchestration flow that coordinates all tasks.

    Args:
        cities: List of cities to process (default: from settings)
        send_notifications: Whether to send notifications

    Returns:
        Pipeline execution summary
    """
    logger = get_run_logger()

    settings = get_settings()

    # Default to configured city
    if cities is None or len(cities) == 0:
        cities = [settings.location.default_city]
        logger.info(f"No cities provided, using default: {cities}")

    logger.info(f"Starting Feature Pipeline for cities: {cities}")
    start_time = datetime.now()

    results = {
        "status": "success",
        "cities_processed": [],
        "cities_failed": [],
        "execution_time": None,
        "errors": [],
    }

    for city in cities:
        try:
            logger.info(f"Processing city: {city}")

            # Step 1: Fetch data
            raw_data = fetch_data(city)

            # Step 2: Save raw data
            filepath = save_raw_data(raw_data, city)

            # Step 3: Create features
            features = create_features(raw_data)

            # Step 4: Store features
            store_features(features)

            results["cities_processed"].append(city)
            logger.info(f"Successfully processed {city}")

        except Exception as e:
            error_msg = f"Failed to process {city}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())

            results["cities_failed"].append(city)
            results["errors"].append(error_msg)

    # Calculate execution time
    results["execution_time"] = str(datetime.now() - start_time)

    # Determine overall status
    if len(results["cities_failed"]) > 0:
        if len(results["cities_processed"]) == 0:
            results["status"] = "failed"
        else:
            results["status"] = "partial"

    # Send notification
    if send_notifications:
        notification_msg = (
            f"Feature Pipeline {results['status'].upper()}\n"
            f"Processed: {len(results['cities_processed'])} cities\n"
            f"Failed: {len(results['cities_failed'])} cities\n"
            f"Time: {results['execution_time']}"
        )
        send_notification(results["status"], notification_msg, results)

    logger.info(f"Feature Pipeline completed with status: {results['status']}")

    return results


# Deployment configuration for scheduling
if __name__ == "__main__":
    # Run pipeline manually
    print("Running Feature Pipeline")
    print("=" * 50)

    results = feature_pipeline(cities=["beijing", "london"], send_notifications=True)

    print("\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Processed: {results['cities_processed']}")
    print(f"  Failed: {results['cities_failed']}")
    print(f"  Time: {results['execution_time']}")
