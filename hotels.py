import requests

url = "https://booking-com.p.rapidapi.com/v1/hotels/search"

querystring = {
    "dest_id": "-2106102",  # New Delhi
    "dest_type": "city",
    "checkin_date": "2025-07-20",
    "checkout_date": "2025-07-27",
    "adults_number": "2",
    "room_number": "1",
    "order_by": "popularity",
    "locale": "en-us",
    "units": "metric",
    "currency": "INR",
    "filter_by_currency": "INR",
    "page_number": "0",
    "include_adjacency": "true"
}


headers = {
    "X-RapidAPI-Key": "5bcd6ecf40mshd2edd4cde23e34ep1bac23jsn61b3c05af993",
    "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

data = response.json()

# ✅ Print basic info for each hotel
for hotel in data.get("result", []):
    print(f"🏨 {hotel.get('hotel_name')}")
    print(f"⭐ Rating: {hotel.get('review_score')}")
    print(f"💰 Price: {hotel.get('price_breakdown', {}).get('gross_price')} {hotel.get('price_breakdown', {}).get('currency')}")
    print(f"📍 Address: {hotel.get('address')}")
    print("🖼️ Image:", hotel.get("main_photo_url"))
    print("-" * 60)
