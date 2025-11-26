import requests
from typing import Dict, Any, Optional
from src.logger import logger
from src.config import Config


class WeatherService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.OPENWEATHER_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.timeout = 10
    
    def get_weather(self, city: str, units: str = "metric") -> Dict[str, Any]:
        try:
            params = {
                "q": city,
                "appid": self.api_key,
                "units": units
            }
            
            logger.info(f"Fetching weather for city: {city}")
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            formatted_data = {
                "city": data.get("name"),
                "country": data.get("sys", {}).get("country"),
                "temperature": data.get("main", {}).get("temp"),
                "feels_like": data.get("main", {}).get("feels_like"),
                "humidity": data.get("main", {}).get("humidity"),
                "pressure": data.get("main", {}).get("pressure"),
                "wind_speed": data.get("wind", {}).get("speed"),
                "wind_direction": data.get("wind", {}).get("deg"),
                "cloudiness": data.get("clouds", {}).get("all"),
                "description": data.get("weather", [{}])[0].get("description"),
                "main_weather": data.get("weather", [{}])[0].get("main"),
                "visibility": data.get("visibility"),
                "sunrise": data.get("sys", {}).get("sunrise"),
                "sunset": data.get("sys", {}).get("sunset"),
            }
            
            logger.info(f"Successfully fetched weather for {city}")
            return formatted_data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"City not found: {city}")
                raise ValueError(f"City '{city}' not found")
            logger.error(f"HTTP error occurred: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}")
            raise
    
    def format_weather_text(self, weather_data: Dict[str, Any]) -> str:
        text = f"""Weather for {weather_data['city']}, {weather_data['country']}:
- Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)
- Condition: {weather_data['description'].capitalize()}
- Humidity: {weather_data['humidity']}%
- Pressure: {weather_data['pressure']} hPa
- Wind Speed: {weather_data['wind_speed']} m/s
- Cloudiness: {weather_data['cloudiness']}%
- Visibility: {weather_data['visibility']} meters"""
        return text.strip()
    
    def validate_api_key(self) -> bool:
        try:
            self.get_weather("London")
            logger.info("OpenWeatherMap API key is valid")
            return True
        except Exception as e:
            logger.warning(f"OpenWeatherMap API key validation failed: {e}")
            return False


weather_service = WeatherService()
