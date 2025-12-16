import sys
import json
import os
import logging

# Suppress logging
logging.basicConfig(level=logging.ERROR)
logging.getLogger("src.utils.logger").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

from src.data.ingestion import fetch_aqi_data
from src.config.settings import get_settings


def debug_fetch():
    print(f"Env API Key: {os.environ.get('AQICN_API_KEY')}")
    settings = get_settings()
    print(f"Settings API Key: {settings.api.aqicn_api_key}")

    city = "Islamabad"
    print(f"Fetching data for {city}...")
    try:
        data = fetch_aqi_data(city)
        print("Fetch successful.")

        if "aqi_data" in data and data["aqi_data"]:
            s = data["aqi_data"].get("station", {})
            print(f"Station: {s.get('name')}")
            print(f"AQI: {data['aqi_data'].get('aqi')}")
        else:
            print("No AQI Data content.")
            print(data)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_fetch()
