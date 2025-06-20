from typing import List, Dict, Optional, Any
from datetime import datetime
import requests
from forex_python.converter import CurrencyRates
import python_weather
import asyncio
import os
from decimal import Decimal
from models import *
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_groq import ChatGroq
from langchain_core.documents import Document

class WeatherService:
    def __init__(self):
        self.api_key = os.getenv('WEATHER_API_KEY')
        if not self.api_key:
            raise ValueError("WEATHER_API_KEY environment variable is not set")
        self.base_url = "https://api.openweathermap.org/data/2.5"

    async def get_current_weather(self, location: Location) -> WeatherInfo:
        url = f"{self.base_url}/weather"
        params = {
            'q': f"{location.city},{location.country}",
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'main' not in data or 'weather' not in data or not data['weather']:
                raise ValueError("Invalid weather data received")
            
            return WeatherInfo(
                temperature=float(data['main']['temp']),
                condition=str(data['weather'][0]['main']),
                humidity=float(data['main']['humidity']),
                wind_speed=float(data['wind']['speed']),
                forecast=[]  # You'll need to make another API call for forecast
            )
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch weather data: {str(e)}")

class AttractionService:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"

    async def search_attractions(self, location: Location) -> List[Attraction]:
        url = f"{self.base_url}/textsearch/json"
        params = {
            'query': f"tourist attractions in {location.city}",
            'key': self.api_key,
            'location': f"{location.coordinates}",
            'radius': '5000'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'results' not in data:
                return []
            
            attractions = []
            for place in data['results']:
                try:
                    if not place or 'name' not in place or 'geometry' not in place:
                        continue
                        
                    attractions.append(Attraction(
                        name=str(place['name']),
                        description=str(place.get('description', '')),
                        rating=float(place.get('rating', 0.0)),
                        price_range=str(place.get('price_level', '$$')),
                        location={
                            'lat': float(place['geometry']['location']['lat']),
                            'lon': float(place['geometry']['location']['lng'])
                        },
                        category=str(place.get('types', ['attraction'])[0])
                    ))
                except (KeyError, ValueError) as e:
                    print(f"Error processing attraction: {str(e)}")
                    continue
            return attractions
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch attractions: {str(e)}")

    async def search_restaurants(self, location: Location) -> List[Restaurant]:
        url = f"{self.base_url}/textsearch/json"
        params = {
            'query': f"restaurants in {location.city}",
            'key': self.api_key,
            'location': f"{location.coordinates}",
            'radius': '5000'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'results' not in data:
                return []
            
            restaurants = []
            for place in data['results']:
                try:
                    if not place or 'name' not in place or 'geometry' not in place:
                        continue
                        
                    restaurants.append(Restaurant(
                        name=str(place['name']),
                        cuisine=str(place.get('types', ['restaurant'])[0]),
                        price_range=str(place.get('price_level', '$$')),
                        rating=float(place.get('rating', 0.0)),
                        location={
                            'lat': float(place['geometry']['location']['lat']),
                            'lon': float(place['geometry']['location']['lng'])
                        }
                    ))
                except (KeyError, ValueError) as e:
                    print(f"Error processing restaurant: {str(e)}")
                    continue
            return restaurants
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch restaurants: {str(e)}")

    async def search_activities(self, location: Location) -> List[Activity]:
        url = f"{self.base_url}/textsearch/json"
        params = {
            'query': f"activities and things to do in {location.city}",
            'key': self.api_key,
            'location': f"{location.coordinates}",
            'radius': '5000'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'results' not in data:
                return []
            
            activities = []
            for place in data['results']:
                try:
                    if not place or 'name' not in place:
                        continue
                        
                    activities.append(Activity(
                        name=str(place['name']),
                        description=str(place.get('description', '')),
                        duration="2-3 hours",  # Default duration
                        price=float(place.get('price_level', 2)) * 20,  # Rough estimate
                        category=str(place.get('types', ['activity'])[0])
                    ))
                except (KeyError, ValueError) as e:
                    print(f"Error processing activity: {str(e)}")
                    continue
            return activities
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch activities: {str(e)}")

class HotelService:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key
        self.base_url = "https://booking-com.p.rapidapi.com/v1"
        self.headers = {
            'x-rapidapi-host': 'booking-com.p.rapidapi.com',
            'x-rapidapi-key': self.api_key
        }

    def get_city_id(self, city_name: str, country: str = None):
        url = f"{self.base_url}/hotels/locations"
        params = {
            'name': city_name,
            'locale': 'en-us'
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            print(f"Booking.com location lookup for {city_name}, {country}: {data}")  # Debug print
            # Find the first city result
            for item in data:
                if item.get('dest_type') == 'city':
                    if country is None or item.get('country') == country or country.lower() in item.get('country', '').lower():
                        return item.get('dest_id')
            # fallback: return first city
            for item in data:
                if item.get('dest_type') == 'city':
                    return item.get('dest_id')
            # Hardcoded fallback for Paris
            if city_name.lower() == "paris":
                return "20088325"
            return None
        except Exception as e:
            print(f"Error fetching city id for {city_name}: {e}")
            # Hardcoded fallback for Paris
            if city_name.lower() == "paris":
                return "20088325"
            return None

    async def search_hotels(self, location: Location, budget_range: tuple) -> List[Hotel]:
        # Get correct city id
        city_id = self.get_city_id(location.city, location.country)
        if not city_id:
            raise Exception(f"Could not find city id for {location.city}, {location.country}")
        url = f"{self.base_url}/hotels/search"
        params = {
            'units': 'metric',
            'room_number': '1',
            'checkout_date': '2024-12-31',
            'checkin_date': '2024-12-30',
            'adults_number': '2',
            'order_by': 'popularity',
            'filter_by_currency': 'USD',
            'locale': 'en-us',
            'dest_type': 'city',
            'dest_id': city_id,
            'page_number': '0',
            'categories_filter_ids': 'class::2,class::4,free_cancellation::1'
        }
        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 422:
                raise Exception(f"Hotel search failed: Invalid city id or parameters for {location.city}, {location.country}")
            response.raise_for_status()
            data = response.json()
            if not data or 'result' not in data:
                return []
            hotels = []
            for hotel in data['result']:
                try:
                    price = float(hotel['price_breakdown']['gross_price'])
                    if budget_range[0] <= price <= budget_range[1]:
                        hotels.append(Hotel(
                            name=str(hotel['hotel_name']),
                            price_per_night=price,
                            rating=float(hotel.get('review_score', 0.0)),
                            amenities=[str(amenity) for amenity in hotel.get('hotel_amenities', [])],
                            location={
                                'lat': float(hotel['latitude']),
                                'lon': float(hotel['longitude'])
                            }
                        ))
                except (KeyError, ValueError) as e:
                    print(f"Error processing hotel: {str(e)}")
                    continue
            return hotels
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch hotels: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to fetch hotels: {str(e)}")

    async def get_hotel_availability(self, hotel_id: str, checkin_date: str, checkout_date: str) -> Dict:
        """Get hotel availability for specific dates"""
        url = f"{self.base_url}/hotels/calendar"
        params = {
            'hotel_id': hotel_id,
            'checkin_date': checkin_date,
            'checkout_date': checkout_date,
            'currency': 'USD',
            'locale': 'en-us'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch hotel availability: {str(e)}")

    async def get_hotel_details(self, hotel_id: str) -> Dict:
        """Get detailed information about a specific hotel"""
        url = f"{self.base_url}/hotels/data"
        params = {
            'hotel_id': hotel_id,
            'locale': 'en-us'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Failed to fetch hotel details: {str(e)}")

    def calculate_hotel_cost(self, hotel: Hotel, nights: int) -> float:
        return float(hotel.price_per_night * nights)

class TransportationService:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key

    async def search_transportation(self, location: Location) -> List[Transportation]:
        # Implement transportation search using appropriate API
        # This is a placeholder implementation
        return []

class CurrencyService:
    def __init__(self):
        self.converter = CurrencyRates()

    def get_exchange_rate(self, from_currency: str, to_currency: str) -> float:
        return float(self.converter.get_rate(from_currency, to_currency))

    def convert_currency(self, amount: float, from_currency: str, to_currency: str) -> float:
        return float(self.converter.convert(from_currency, to_currency, amount))

def travel_plan_to_documents(travel_plan: TravelPlan) -> list:
    docs = []
    # Add summary
    docs.append(Document(page_content=travel_plan.summary, metadata={"type": "summary"}))
    # Add weather
    weather = travel_plan.weather_forecast
    weather_text = f"Weather: {weather.condition}, Temp: {weather.temperature}C, Humidity: {weather.humidity}%, Wind: {weather.wind_speed}kph"
    docs.append(Document(page_content=weather_text, metadata={"type": "weather"}))
    # Add attractions
    for attr in travel_plan.attractions:
        content = f"Attraction: {attr.name}\n{attr.description}\nRating: {attr.rating}, Price: {attr.price_range}, Category: {attr.category}, Location: {attr.location}"
        docs.append(Document(page_content=content, metadata={"type": "attraction", "name": attr.name}))
    # Add restaurants
    for rest in travel_plan.restaurants:
        content = f"Restaurant: {rest.name}\nCuisine: {rest.cuisine}, Price: {rest.price_range}, Rating: {rest.rating}, Location: {rest.location}"
        docs.append(Document(page_content=content, metadata={"type": "restaurant", "name": rest.name}))
    # Add hotels
    for hotel in travel_plan.hotels:
        content = f"Hotel: {hotel.name}\nPrice per night: {hotel.price_per_night}, Rating: {hotel.rating}, Amenities: {hotel.amenities}, Location: {hotel.location}"
        docs.append(Document(page_content=content, metadata={"type": "hotel", "name": hotel.name}))
    # Add itinerary days
    for day in travel_plan.itinerary.daily_plans:
        day_content = f"Day: {day.date.strftime('%Y-%m-%d')}\n"
        day_content += "Activities:\n" + "\n".join([f"- {a.name}: {a.description} ({a.duration}, {a.category}, ${a.price})" for a in day.activities])
        day_content += "\nMeals:\n" + "\n".join([f"- {m.name}: {m.cuisine} ({m.price_range})" for m in day.meals])
        docs.append(Document(page_content=day_content, metadata={"type": "itinerary_day", "date": day.date.strftime('%Y-%m-%d')}))
    # Add budget
    budget = travel_plan.budget
    budget_text = f"Budget: Total {budget.total_budget} {budget.currency}, Daily {budget.daily_budget} {budget.currency}, Expenses: {budget.expenses}"
    docs.append(Document(page_content=budget_text, metadata={"type": "budget"}))
    return docs

class ItineraryService:
    def __init__(self):
        print("ItineraryService....0")
        self.weather_service = WeatherService()
        print("ItineraryService....1")
        self.attraction_service = AttractionService(os.getenv('ATTRACTION_API_KEY', ''))
        print("ItineraryService....2")
        self.hotel_service = HotelService(os.getenv('HOTEL_API_KEY', ''))
        print("ItineraryService....3")
        #self.transportation_service = TransportationService(os.getenv('TRANSPORTATION_API_KEY', ''))
        self.currency_service = CurrencyService()
        # Pinecone embedding service
        self.pinecone_service = PineconeEmbeddingService()

    async def generate_itinerary(self, location: Location, start_date: datetime, 
                               end_date: datetime, budget: float, currency: str) -> TravelPlan:
        try:
            # Generate complete itinerary
            weather = await self.weather_service.get_current_weather(location)
            attractions = await self.attraction_service.search_attractions(location)
            restaurants = await self.attraction_service.search_restaurants(location)
            activities = await self.attraction_service.search_activities(location)
            hotels = await self.hotel_service.search_hotels(location, (budget * 0.3, budget * 0.5))
            #transportation = await self.transportation_service.search_transportation(location)

            # Create daily plans
            daily_plans = []
            current_date = start_date
            while current_date <= end_date:
                daily_plan = DailyPlan(
                    date=current_date,
                    activities=activities[:2],  # Example: 2 activities per day
                    meals=restaurants[:3],      # Example: 3 meals per day
                    transportation=[]#transportation[:2]  # Example: 2 transportation options
                )
                daily_plans.append(daily_plan)
                current_date = current_date.replace(day=current_date.day + 1)

            # Calculate total cost
            total_cost = sum(
                float(hotel.price_per_night * (end_date - start_date).days)
                for hotel in hotels
            )

            # Create itinerary
            itinerary = Itinerary(
                location=location,
                start_date=start_date,
                end_date=end_date,
                daily_plans=daily_plans,
                total_cost=total_cost,
                currency=currency
            )

            # Create budget
            budget_obj = Budget(
                total_budget=budget,
                daily_budget=budget / (end_date - start_date).days,
                currency=currency,
                expenses={
                    'hotels': total_cost,
                    'activities': sum(float(activity.price) for activity in activities),
                    'meals': sum(float(restaurant.price_range) for restaurant in restaurants),
                    #'transportation': sum(float(trans.cost) for trans in transportation)
                }
            )

            # Generate summary
            summary = f"Travel plan for {location.city}, {location.country}\n"
            summary += f"Duration: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
            summary += f"Total budget: {budget} {currency}\n"
            summary += f"Daily budget: {budget_obj.daily_budget} {currency}\n"
            summary += f"Number of attractions: {len(attractions)}\n"
            summary += f"Number of restaurants: {len(restaurants)}\n"
            summary += f"Number of activities: {len(activities)}"

            travel_plan = TravelPlan(
                itinerary=itinerary,
                weather_forecast=weather,
                budget=budget_obj,
                attractions=attractions,
                restaurants=restaurants,
                hotels=hotels,
                summary=summary
            )

            # === Embed all travel info into Pinecone ===
            docs = travel_plan_to_documents(travel_plan)
            self.pinecone_service.add_documents(docs)

            return travel_plan
        except Exception as e:
            raise Exception(f"Failed to generate itinerary: {str(e)}")

# === Pinecone Vector Embedding and Groq LLM Integration ===
class PineconeEmbeddingService:
    def __init__(self, index_name="travel-agency-index", dimension=1536, region="us-east-1", cloud="aws"):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        self.region = region
        self.cloud = cloud
        if not self.pc.has_index(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self.index = self.pc.Index(index_name)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)

    def add_documents(self, documents, ids=None):
        """documents: list of langchain_core.documents.Document"""
        from uuid import uuid4
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=ids)
        return ids

    def similarity_search(self, query, k=2, filter=None):
        return self.vector_store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(self, query, k=2, filter=None):
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)

class GroqLLMService:
    def __init__(self, model_name="deepseek-r1-distill-llama-70b", temperature=0):
        self.llm = ChatGroq(
            model_name=model_name,
            temperature=temperature
        )
    def invoke(self, prompt):
        return self.llm.invoke(prompt)