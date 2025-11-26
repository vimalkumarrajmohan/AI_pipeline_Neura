import unittest
from unittest.mock import patch, MagicMock
from src.weather_service import WeatherService


class TestWeatherService(unittest.TestCase):
    def setUp(self):
        self.service = WeatherService(api_key="test_api_key")
        self.mock_response = {
            "name": "London",
            "sys": {"country": "GB"},
            "main": {
                "temp": 15.5,
                "feels_like": 14.2,
                "humidity": 72,
                "pressure": 1013
            },
            "wind": {"speed": 4.5, "deg": 230},
            "clouds": {"all": 60},
            "weather": [{"main": "Clouds", "description": "Partly cloudy"}],
            "visibility": 10000,
            "cod": "200"
        }
    
    @patch('src.weather_service.requests.get')
    def test_get_weather_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_response
        mock_get.return_value = mock_response
        
        result = self.service.get_weather("London")
        
        self.assertEqual(result["city"], "London")
        self.assertEqual(result["country"], "GB")
        self.assertEqual(result["temperature"], 15.5)
        self.assertEqual(result["humidity"], 72)
    
    @patch('src.weather_service.requests.get')
    def test_get_weather_city_not_found(self, mock_get):
        import requests
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError):
            self.service.get_weather("NonExistentCity")
    
    @patch('src.weather_service.requests.get')
    def test_get_weather_api_error(self, mock_get):
        mock_get.side_effect = Exception("API Error")
        
        with self.assertRaises(Exception):
            self.service.get_weather("London")
    
    def test_format_weather_text(self):
        weather_data = {
            "city": "London",
            "country": "GB",
            "temperature": 15.5,
            "feels_like": 14.2,
            "humidity": 72,
            "pressure": 1013,
            "wind_speed": 4.5,
            "cloudiness": 60,
            "description": "Partly cloudy",
            "visibility": 10000
        }
        
        text = self.service.format_weather_text(weather_data)
        
        self.assertIn("London", text)
        self.assertIn("15.5", text)
        self.assertIn("Partly cloudy", text)
        self.assertIn("72%", text)
    
    @patch('src.weather_service.requests.get')
    def test_validate_api_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_response
        mock_get.return_value = mock_response
        
        result = self.service.validate_api_key()
        
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
