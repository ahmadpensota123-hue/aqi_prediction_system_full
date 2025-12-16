"""
Data Ingestion Module
=====================

This module handles fetching data from external APIs:
1. AQICN (World Air Quality Index) - Real-time and historical AQI data
2. OpenWeatherMap - Weather data (temperature, humidity, wind, etc.)

Why these APIs?
- AQICN provides global air quality data with standardized AQI values
- OpenWeatherMap offers free tier with weather forecasts
- Both have good documentation and reliability

Data Flow:
1. Create API clients with authentication
2. Fetch data for specified locations
3. Parse and validate responses
4. Store raw data to disk for reproducibility
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.settings import get_settings, get_aqi_category
from src.utils.logger import get_logger, log_execution_time, log_exception

logger = get_logger(__name__)


class BaseAPIClient:
    """
    Base class for API clients with retry logic and error handling.
    
    Features:
    - Automatic retries with exponential backoff
    - Rate limiting handling
    - Response validation
    - Logging of all requests
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the API client.
        
        Args:
            base_url: The base URL for the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """
        Create a requests session with retry logic.
        
        Retry strategy:
        - 3 retries total
        - Exponential backoff (1s, 2s, 4s)
        - Retry on 500, 502, 503, 504 errors
        """
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        method: str = "GET"
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with error handling.
        
        Args:
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            method: HTTP method (GET, POST, etc.)
        
        Returns:
            Parsed JSON response
        
        Raises:
            requests.RequestException: On network errors
            ValueError: On invalid responses
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        logger.debug(f"Making {method} request to {url}")
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from {url}")
            raise ValueError("Invalid JSON response from API")


class AQICNClient(BaseAPIClient):
    """
    Client for the World Air Quality Index (AQICN) API.
    
    API Documentation: https://aqicn.org/api/
    
    The AQICN API provides:
    - Real-time AQI data for cities worldwide
    - Pollutant concentrations (PM2.5, PM10, O3, NO2, SO2, CO)
    - Weather data (temperature, humidity, pressure, wind)
    
    Free tier includes:
    - 1000 requests/day
    - Real-time data only (no historical)
    - "demo" token for testing
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize AQICN client.
        
        Args:
            api_key: AQICN API token. 
                     Get yours at: https://aqicn.org/data-platform/token/
                     Use "demo" for testing (limited functionality)
        """
        settings = get_settings()
        self.api_key = api_key or settings.api.aqicn_api_key
        super().__init__(settings.api.aqicn_base_url)
        
        if self.api_key == "demo":
            logger.warning(
                "Using demo API key. Get a real key at: "
                "https://aqicn.org/data-platform/token/"
            )
    
    @log_execution_time
    def get_city_aqi(self, city: str) -> Dict[str, Any]:
        """
        Get current AQI data for a city.
        
        Args:
            city: City name (e.g., "beijing", "london", "new york")
        
        Returns:
            Dict containing:
            - aqi: Overall AQI value
            - pollutants: Individual pollutant values
            - weather: Weather conditions
            - timestamp: Data timestamp
        
        Example:
            >>> client = AQICNClient()
            >>> data = client.get_city_aqi("beijing")
            >>> print(f"AQI: {data['aqi']}")
        """
        response = self._make_request(
            f"feed/{city}/",
            params={"token": self.api_key}
        )
        
        if response.get("status") != "ok":
            error_msg = response.get("data", "Unknown error")
            logger.error(f"AQICN API error for {city}: {error_msg}")
            raise ValueError(f"AQICN API error: {error_msg}")
        
        return self._parse_aqi_response(response["data"])
    
    @log_execution_time
    def get_geo_aqi(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Get current AQI data for geographic coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            Dict containing AQI data for nearest station
        """
        response = self._make_request(
            f"feed/geo:{lat};{lon}/",
            params={"token": self.api_key}
        )
        
        if response.get("status") != "ok":
            error_msg = response.get("data", "Unknown error")
            logger.error(f"AQICN API error for ({lat}, {lon}): {error_msg}")
            raise ValueError(f"AQICN API error: {error_msg}")
        
        return self._parse_aqi_response(response["data"])
    
    def search_city(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search for cities by keyword.
        
        Args:
            keyword: Search term
        
        Returns:
            List of matching cities with their UIDs and names
        """
        response = self._make_request(
            "search/",
            params={"token": self.api_key, "keyword": keyword}
        )
        
        if response.get("status") != "ok":
            return []
        
        return [
            {
                "uid": station.get("uid"),
                "name": station.get("station", {}).get("name"),
                "time": station.get("time", {}).get("stime")
            }
            for station in response.get("data", [])
        ]
    
    def _parse_aqi_response(self, data: Dict) -> Dict[str, Any]:
        """
        Parse and structure AQICN API response.
        
        Extracts:
        - Overall AQI
        - Individual pollutants (PM2.5, PM10, O3, NO2, SO2, CO)
        - Weather data (temp, humidity, pressure, wind)
        - Location info
        """
        aqi_value = data.get("aqi", 0)
        
        # Handle "aqi": "-" (no data available)
        if isinstance(aqi_value, str):
            aqi_value = 0
        
        # Extract pollutants
        iaqi = data.get("iaqi", {})
        pollutants = {
            "pm25": iaqi.get("pm25", {}).get("v"),
            "pm10": iaqi.get("pm10", {}).get("v"),
            "o3": iaqi.get("o3", {}).get("v"),
            "no2": iaqi.get("no2", {}).get("v"),
            "so2": iaqi.get("so2", {}).get("v"),
            "co": iaqi.get("co", {}).get("v"),
        }
        
        # Extract weather
        weather = {
            "temperature": iaqi.get("t", {}).get("v"),
            "humidity": iaqi.get("h", {}).get("v"),
            "pressure": iaqi.get("p", {}).get("v"),
            "wind": iaqi.get("w", {}).get("v"),
        }
        
        # Get category info
        category_info = get_aqi_category(aqi_value)
        
        # Parse timestamp
        time_data = data.get("time", {})
        timestamp = time_data.get("iso") or datetime.now().isoformat()
        
        result = {
            "aqi": aqi_value,
            "category": category_info["category"],
            "color": category_info["color"],
            "health_message": category_info["health_message"],
            "pollutants": pollutants,
            "weather": weather,
            "station": {
                "name": data.get("city", {}).get("name"),
                "latitude": data.get("city", {}).get("geo", [None, None])[0],
                "longitude": data.get("city", {}).get("geo", [None, None])[1],
            },
            "timestamp": timestamp,
            "dominant_pollutant": data.get("dominentpol"),
            "attributions": data.get("attributions", []),
        }
        
        logger.info(
            f"Retrieved AQI {aqi_value} ({category_info['category']}) "
            f"for {result['station']['name']}"
        )
        
        return result


class OpenWeatherClient(BaseAPIClient):
    """
    Client for the OpenWeatherMap API.
    
    API Documentation: https://openweathermap.org/api
    
    The OpenWeatherMap API provides:
    - Current weather conditions
    - 5-day forecast (free tier)
    - Historical weather data (paid tier)
    
    Free tier includes:
    - 1000 calls/day
    - Current weather
    - 5-day/3-hour forecast
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenWeatherMap client.
        
        Args:
            api_key: OpenWeather API key.
                     Sign up at: https://openweathermap.org/api
        """
        settings = get_settings()
        self.api_key = api_key or settings.api.openweather_api_key
        super().__init__(settings.api.openweather_base_url)
        
        if not self.api_key:
            logger.warning(
                "OpenWeather API key not set. "
                "Get one at: https://openweathermap.org/api"
            )
    
    @log_execution_time
    def get_current_weather(
        self,
        city: str = None,
        lat: float = None,
        lon: float = None,
        units: str = "metric"
    ) -> Dict[str, Any]:
        """
        Get current weather for a location.
        
        Args:
            city: City name (e.g., "London", "Beijing")
            lat: Latitude (use with lon)
            lon: Longitude (use with lat)
            units: "metric" (Celsius), "imperial" (Fahrenheit), or "kelvin"
        
        Returns:
            Dict containing weather data
        """
        if not self.api_key:
            raise ValueError("OpenWeather API key is required")
        
        params = {
            "appid": self.api_key,
            "units": units
        }
        
        if city:
            params["q"] = city
        elif lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        else:
            raise ValueError("Either city or lat/lon must be provided")
        
        response = self._make_request("weather", params=params)
        
        return self._parse_weather_response(response)
    
    @log_execution_time
    def get_forecast(
        self,
        city: str = None,
        lat: float = None,
        lon: float = None,
        units: str = "metric"
    ) -> List[Dict[str, Any]]:
        """
        Get 5-day weather forecast (3-hour intervals).
        
        Args:
            city: City name
            lat: Latitude
            lon: Longitude
            units: Temperature units
        
        Returns:
            List of forecast entries (up to 40 entries = 5 days)
        """
        if not self.api_key:
            raise ValueError("OpenWeather API key is required")
        
        params = {
            "appid": self.api_key,
            "units": units
        }
        
        if city:
            params["q"] = city
        elif lat is not None and lon is not None:
            params["lat"] = lat
            params["lon"] = lon
        else:
            raise ValueError("Either city or lat/lon must be provided")
        
        response = self._make_request("forecast", params=params)
        
        forecasts = []
        for entry in response.get("list", []):
            forecasts.append(self._parse_forecast_entry(entry))
        
        return forecasts
    
    def _parse_weather_response(self, data: Dict) -> Dict[str, Any]:
        """Parse current weather response."""
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        
        return {
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "temp_min": main.get("temp_min"),
            "temp_max": main.get("temp_max"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "visibility": data.get("visibility"),
            "clouds": data.get("clouds", {}).get("all"),
            "weather_condition": weather.get("main"),
            "weather_description": weather.get("description"),
            "location": {
                "city": data.get("name"),
                "country": data.get("sys", {}).get("country"),
                "latitude": data.get("coord", {}).get("lat"),
                "longitude": data.get("coord", {}).get("lon"),
            },
            "timestamp": datetime.fromtimestamp(data.get("dt", 0)).isoformat(),
        }
    
    def _parse_forecast_entry(self, data: Dict) -> Dict[str, Any]:
        """Parse a single forecast entry."""
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        
        return {
            "timestamp": data.get("dt_txt"),
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "temp_min": main.get("temp_min"),
            "temp_max": main.get("temp_max"),
            "humidity": main.get("humidity"),
            "pressure": main.get("pressure"),
            "wind_speed": wind.get("speed"),
            "wind_direction": wind.get("deg"),
            "clouds": data.get("clouds", {}).get("all"),
            "weather_condition": weather.get("main"),
            "weather_description": weather.get("description"),
            "pop": data.get("pop"),  # Probability of precipitation
        }


class DataIngestionService:
    """
    High-level service for data ingestion.
    
    This service coordinates data fetching from multiple APIs
    and handles data storage.
    
    Usage:
        >>> service = DataIngestionService()
        >>> data = service.fetch_all_data("beijing")
        >>> service.save_raw_data(data, "beijing")
    """
    
    def __init__(self):
        """Initialize the data ingestion service."""
        self.settings = get_settings()
        self.aqicn_client = AQICNClient()
        self.openweather_client = OpenWeatherClient()
        
        # Create data directories
        self.raw_data_dir = Path(self.settings.app.raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    @log_execution_time
    @log_exception
    def fetch_all_data(self, city: str = None) -> Dict[str, Any]:
        """
        Fetch all available data for a city.
        
        Combines:
        - Real-time AQI data from AQICN
        - Current weather from OpenWeather
        - Weather forecast from OpenWeather
        
        Args:
            city: City name (defaults to settings.default_city)
        
        Returns:
            Combined data dictionary
        """
        city = city or self.settings.location.default_city
        
        logger.info(f"Fetching all data for city: {city}")
        
        result = {
            "city": city,
            "fetch_timestamp": datetime.now().isoformat(),
            "aqi_data": None,
            "current_weather": None,
            "forecast": None,
        }
        
        # Fetch AQI data
        try:
            result["aqi_data"] = self.aqicn_client.get_city_aqi(city)
        except Exception as e:
            logger.error(f"Failed to fetch AQI data: {e}")
            result["aqi_data"] = {"error": str(e)}
        
        # Fetch weather data (only if API key is available)
        if self.openweather_client.api_key:
            try:
                result["current_weather"] = self.openweather_client.get_current_weather(city)
            except Exception as e:
                logger.error(f"Failed to fetch current weather: {e}")
                result["current_weather"] = {"error": str(e)}
            
            try:
                result["forecast"] = self.openweather_client.get_forecast(city)
            except Exception as e:
                logger.error(f"Failed to fetch forecast: {e}")
                result["forecast"] = {"error": str(e)}
        else:
            logger.warning("Skipping weather data - no OpenWeather API key")
        
        return result
    
    def save_raw_data(
        self,
        data: Dict[str, Any],
        city: str,
        timestamp: datetime = None
    ) -> Path:
        """
        Save raw data to a JSON file.
        
        Files are organized by date and city:
        data/raw/2024-01-15/beijing_143022.json
        
        Args:
            data: Data to save
            city: City name
            timestamp: Timestamp for filename (defaults to now)
        
        Returns:
            Path to saved file
        """
        timestamp = timestamp or datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H%M%S")
        
        # Create date directory
        date_dir = self.raw_data_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        filename = f"{city.lower().replace(' ', '_')}_{time_str}.json"
        filepath = date_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved raw data to {filepath}")
        
        return filepath
    
    def load_raw_data(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load raw data from a JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            Loaded data dictionary
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def fetch_multiple_cities(self, cities: List[str]) -> Dict[str, Dict]:
        """
        Fetch data for multiple cities.
        
        Args:
            cities: List of city names
        
        Returns:
            Dict mapping city names to their data
        """
        results = {}
        
        for city in cities:
            try:
                logger.info(f"Fetching data for {city}...")
                results[city] = self.fetch_all_data(city)
                
                # Rate limiting - be nice to APIs
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {city}: {e}")
                results[city] = {"error": str(e)}
        
        return results


# Convenience function for quick data fetching
def fetch_aqi_data(city: str = None) -> Dict[str, Any]:
    """
    Quick function to fetch AQI data for a city.
    
    Args:
        city: City name (defaults to config default)
    
    Returns:
        AQI data dictionary
    
    Example:
        >>> data = fetch_aqi_data("beijing")
        >>> print(f"AQI: {data['aqi']}, Category: {data['category']}")
    """
    service = DataIngestionService()
    return service.fetch_all_data(city)


if __name__ == "__main__":
    # Test the data ingestion
    print("Testing Data Ingestion Module")
    print("=" * 50)
    
    # Test AQICN client
    print("\n1. Testing AQICN API (demo key):")
    aqicn = AQICNClient()
    
    try:
        data = aqicn.get_city_aqi("beijing")
        print(f"   City: {data['station']['name']}")
        print(f"   AQI: {data['aqi']}")
        print(f"   Category: {data['category']}")
        print(f"   PM2.5: {data['pollutants']['pm25']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test city search
    print("\n2. Testing City Search:")
    results = aqicn.search_city("new york")
    for station in results[:3]:
        print(f"   - {station['name']}")
    
    # Test full data ingestion
    print("\n3. Testing Full Data Ingestion:")
    service = DataIngestionService()
    all_data = service.fetch_all_data("london")
    
    if all_data.get("aqi_data") and "error" not in all_data["aqi_data"]:
        print(f"   AQI Data: Retrieved successfully")
    
    # Save sample data
    saved_path = service.save_raw_data(all_data, "london")
    print(f"   Saved to: {saved_path}")
    
    print("\n" + "=" * 50)
    print("Data Ingestion Test Complete!")
