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
        fetched_data = requests.get(base_url, params=params)
        fetched_data.raise_for_status()
        weather_data = fetched_data.json()

        # Extract all available weather information
        city_name = weather_data['name']
        country = weather_data['sys']['country']
        
        # Temperature data
        temp = weather_data['main']['temp']
        feels_like = weather_data['main']['feels_like']
        temp_min = weather_data['main']['temp_min']
        temp_max = weather_data['main']['temp_max']
        
        # Weather condition
        condition = weather_data['weather'][0]['description']
        main_weather = weather_data['weather'][0]['main']
        
        # Atmospheric data
        pressure = weather_data['main']['pressure']
        humidity = weather_data['main']['humidity']
        
        # Wind data
        wind_speed = weather_data['wind']['speed']
        wind_deg = weather_data['wind'].get('deg', 'N/A')
        
        # Visibility
        visibility = weather_data.get('visibility', 'N/A')
        if visibility != 'N/A':
            visibility = visibility / 1000  # Convert to km
        
        # Cloudiness
        cloudiness = weather_data['clouds']['all']
        
        # Sunrise and sunset
        from datetime import datetime
        sunrise = datetime.fromtimestamp(weather_data['sys']['sunrise']).strftime('%H:%M:%S')
        sunset = datetime.fromtimestamp(weather_data['sys']['sunset']).strftime('%H:%M:%S')
        
        # Coordinates
        lon = weather_data['coord']['lon']
        lat = weather_data['coord']['lat']

        # Display all weather data
        print(f"\n{'='*50}")
        print(f"Weather Report for {city_name}, {country}")
        print(f"{'='*50}")
        
        print(f"\n📍 Location:")
        print(f"   Coordinates: {lat}°N, {lon}°E")
        
        print(f"\n🌡️  Temperature:")
        print(f"   Current: {temp}°C")
        print(f"   Feels Like: {feels_like}°C")
        print(f"   Min: {temp_min}°C")
        print(f"   Max: {temp_max}°C")
        
        print(f"\n🌤️  Weather Condition:")
        print(f"   Main: {main_weather}")
        print(f"   Description: {condition.capitalize()}")
        
        print(f"\n💨 Wind:")
        print(f"   Speed: {wind_speed} m/s")
        print(f"   Direction: {wind_deg}°")
        
        print(f"\n🌫️  Atmospheric Conditions:")
        print(f"   Pressure: {pressure} hPa")
        print(f"   Humidity: {humidity}%")
        print(f"   Cloudiness: {cloudiness}%")
        print(f"   Visibility: {visibility} km" if visibility != 'N/A' else f"   Visibility: {visibility}")
        
        print(f"\n🌅 Sun:")
        print(f"   Sunrise: {sunrise}")
        print(f"   Sunset: {sunset}")
        
        print(f"\n{'='*50}\n")

    except requests.exceptions.HTTPError as http_err:
        if fetched_data.status_code == 404:
            print(f"Error: City '{city}' not found.")
        elif fetched_data.status_code == 401:
            print("Error: Invalid API key.")
        else:
            print(f"HTTP error: {http_err}")
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Check your internet.")
    except Exception as err:
        print(f"Error: {err}")


city = input("Enter city name: ").strip()
if city:
    get_weather(city)
else:
    print("Error: City name is required.")
