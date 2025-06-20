from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class Location(BaseModel):
    city: str
    country: str
    coordinates: Optional[Dict[str, float]] = None

class WeatherInfo(BaseModel):
    temperature: float
    condition: str
    humidity: float
    wind_speed: float
    forecast: List[Dict] = Field(default_factory=list)

class Attraction(BaseModel):
    name: str
    description: str
    rating: float
    price_range: str
    location: Dict[str, float]
    category: str

class Restaurant(BaseModel):
    name: str
    cuisine: str
    price_range: str
    rating: float
    location: Dict[str, float]

class Activity(BaseModel):
    name: str
    description: str
    duration: str
    price: float
    category: str

class Hotel(BaseModel):
    name: str
    price_per_night: float
    rating: float
    amenities: List[str]
    location: Dict[str, float]

class Transportation(BaseModel):
    type: str
    cost: float
    duration: str
    provider: str

class DailyPlan(BaseModel):
    date: datetime
    activities: List[Activity]
    meals: List[Restaurant]
    transportation: List[Transportation]

class Itinerary(BaseModel):
    location: Location
    start_date: datetime
    end_date: datetime
    daily_plans: List[DailyPlan]
    total_cost: float
    currency: str

class Budget(BaseModel):
    total_budget: float
    daily_budget: float
    currency: str
    expenses: Dict[str, float] = Field(default_factory=dict)

class TravelPlan(BaseModel):
    itinerary: Itinerary
    weather_forecast: WeatherInfo
    budget: Budget
    attractions: List[Attraction]
    restaurants: List[Restaurant]
    hotels: List[Hotel]
    summary: str 