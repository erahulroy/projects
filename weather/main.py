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
        # fetched_data stores http response object like:
        #   fetched_data.status_code: 200 if data found, 404 if no city found, 401 if invalid API key
        #   fetched_data.text: received data in text formate (here we receive in JSON (plain text formated like dictionaries) formate)
        # if there is no internet connection then it raises requests.exceptions.ConnectionError
        fetched_data = requests.get(base_url, params=params)
        # .raise_for_status() method is used to check http status code. if code is 200-299, successful fetch and does nothing. if code is 400-499, means client error and raises requests.exceptions.HTTPError. if code is 500-599, means server error and raises requests.exceptions.HTTPError.
        fetched_data.raise_for_status()
        # .json() method used to convert received JSON to python dictionaries
        weather_data = fetched_data.json()

        city_name = weather_data['name']
        temperature = weather_data['main']['temp']
        condition = weather_data['weather'][0]['description']

        print(f"\nWeather in {city_name}")
        print(f"Temperature: {temperature}°C")
        print(f"Condition: {condition}")

    except requests.exceptions.HTTPError as http_err:
        if fetched_data.status_code == 404:
            print(f"Error: City '{city}' not found.")
        elif fetched_data.status_code == 401:
            print("Error: Invalid API key.")
        else:
            print(f"HTTP error: {http_err}")
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Check your internet.")
    # catch-all other exceptions if any occurs
    except Exception as err:
        print(f"Error: {err}")


city = input("Enter city name: ").strip()
if city:
    get_weather(city)
else:
    print("Error: City name is required.")
