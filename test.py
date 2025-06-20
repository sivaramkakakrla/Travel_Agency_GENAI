import os
from dotenv import load_dotenv
import requests

load_dotenv()

def test_weather_api():
    api_key = os.getenv('WEATHER_API_KEY')
    url = f"https://api.openweathermap.org/data/2.5/weather?q=London&appid={api_key}"
    response = requests.get(url)
    return response.status_code == 200

def test_places_api():
    api_key = os.getenv('ATTRACTION_API_KEY')
    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query=restaurants+in+London&key={api_key}"
    response = requests.get(url)
    return response.status_code == 200

def test_hotel_api():
    api_key = os.getenv('HOTEL_API_KEY')
    
    url = "https://booking-com.p.rapidapi.com/v1/attractions/calendar"
    querystring = {
    "attraction_id": "PRFZkGSVnM5d",
    "currency": "AED",
    "locale": "en-gb"
    }

    headers = {
    "x-rapidapi-host": "booking-com.p.rapidapi.com",
    "x-rapidapi-key": "5bcd6ecf40mshd2edd4cde23e34ep1bac23jsn61b3c05af993"
    }

    response = requests.get(url, headers=headers, params=querystring)
    print("hotel api test")
    print(response.status_code)
    return response.status_code == 200

print("Testing API Keys:")
print(f"Weather API: {'✓' if test_weather_api() else '✗'}")
print(f"Places API: {'✓' if test_places_api() else '✗'}")
print(f"Hotel API: {'✓' if test_hotel_api() else '✗'}")