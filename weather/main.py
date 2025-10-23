import requests

class WeatherFetcher:
    def __init__(self, city):
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"
        self.api_key = "a612116ff3018aa4cae2884314f84e49"
        self.city = city

    def fetch_weather(self):
        params = {
            'q': self.city,
            'appid': self.api_key,
            'units': 'metric'
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404:
                print(f"Error: City '{self.city}' not found.")
            elif response.status_code == 401:
                print("Error: Invalid API key.")
            else:
                print(f"HTTP error: {http_err}")
        except requests.exceptions.ConnectionError:
            print("Error: Connection failed. Check your internet.")
        except Exception as err:
            print(f"Unexpected error: {err}")
        return None

    def display_weather(self, weather_data):
        try:
            city_name = weather_data['name']
            temperature = weather_data['main']['temp']
            condition = weather_data['weather'][0]['description']

            print(f"\nWeather in {city_name}")
            print(f"Temperature: {temperature}°C")
            print(f"Condition: {condition}")
        except KeyError as key_err:
            print(f"Error: Missing data in API response: {key_err}")

    def run(self):
        weather_data = self.fetch_weather()
        if weather_data:
            self.display_weather(weather_data)


city = input("Enter city name (e.g., London, New York, Tokyo): ").strip()

if city:
    WeatherFetcher(city).run()
else:
    print("Error: City name is required.")
