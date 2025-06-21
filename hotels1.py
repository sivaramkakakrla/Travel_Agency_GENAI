import requests

url = "https://booking-com.p.rapidapi.com/v1/hotels/search"

param = {
    "units": "metric",
    "room_number": "1",
    "checkout_date": "2025-07-13",
    "checkin_date": "2025-07-12",
    "adults_number": "2",
    "locale": "en-us",
    "dest_type": "city",
    "dest_id": "-2101353",  # Mumbai
    "page_number": "0",
    "order_by": "popularity",
    "currency": "INR",
    "filter_by_currency": "INR",  # ✅ Required field
    "include_adjacency": "true"
}



headers = {
    "X-RapidAPI-Key": "5bcd6ecf40mshd2edd4cde23e34ep1bac23jsn61b3c05af993",
    "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=param)
#response.raise_for_status()
data = response.json()
hotels = data.get("result", [])

print("Status:", response.status_code)
print("URL:", response.url)
print("Raw Response:", response.text)


print("11111111111111111")
for hotel in hotels:
    print("22222222222222")
    print(f"🏨 Hotel: {hotel.get('hotel_name')}")
    print(f"⭐ Rating: {hotel.get('review_score')}")
    print(f"💰 Price: {hotel.get('price_breakdown', {}).get('gross_price')} {hotel.get('price_breakdown', {}).get('currency')}")
    print(f"📍 Address: {hotel.get('address')}")
    print(f"🖼️ Image: {hotel.get('main_photo_url')}")
    print("-" * 60)
print("33333333333")