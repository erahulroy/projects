import requests

def get_weather(city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    api_key = "a612116ff3018aa4cae2884314f84e49"

    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        weather_data = response.json()
        print(weather_data)

        city_name = weather_data['name']
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description']

        print(f"\nWeather in {city_name}")
        print(f"Temperature: {temperature}°C")
        print(f"Condition: {condition}")

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            print(f"Error: City '{city}' not found.")
        elif response.status_code == 401:
            print("Error: Invalid API key.")
        else:
            print(f"HTTP error: {http_err}")
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Check your internet.")
    except KeyError as key_err:
        print(f"Error: Missing data in API response: {key_err}")
    except Exception as err:
        print(f"Unexpected error: {err}")

city = input("Enter city name (e.g., London, New York, Tokyo): ").strip()
if city:
    get_weather(city)
else:
    print("Error: City name is required.")
