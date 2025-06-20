from typing import Dict, List, Tuple, Any
from langgraph.graph import Graph, StateGraph
from datetime import datetime
from models import *
from services import ItineraryService

class TravelAgent:
    def __init__(self):
        print("TravelAgent....0")
        self.itinerary_service = ItineraryService()
        print("TravelAgent....00")
        self.graph = self._build_graph()
        print("TravelAgent....1")

    def _build_graph(self) -> Any:
        print("_build_graph....1")
        # Define the nodes
        def init_state(state: dict) -> dict:
            state["current_step"] = "search_attractions"
            return state

        async def search_attractions(state: dict) -> dict:
            if not state.get("location"):
                raise ValueError("Location not set")
            state["attractions"] = await self.itinerary_service.attraction_service.search_attractions(state["location"])
            state["restaurants"] = await self.itinerary_service.attraction_service.search_restaurants(state["location"])
            state["activities"] = await self.itinerary_service.attraction_service.search_activities(state["location"])
            state["current_step"] = "search_weather"
            return state

        async def search_weather(state: dict) -> dict:
            if not state.get("location"):
                raise ValueError("Location not set")
            state["weather"] = await self.itinerary_service.weather_service.get_current_weather(state["location"])
            state["current_step"] = "search_hotels"
            return state

        async def search_hotels(state: dict) -> dict:
            if not state.get("location") or not state.get("budget"):
                raise ValueError("Location or budget not set")
            state["hotels"] = await self.itinerary_service.hotel_service.search_hotels(
                state["location"],
                (state["budget"] * 0.3, state["budget"] * 0.5)
            )
            state["current_step"] = "calculate_costs"
            return state

        async def calculate_costs(state: dict) -> dict:
            if not all([state.get("location"), state.get("start_date"), state.get("end_date"), state.get("budget")]):
                raise ValueError("Missing required information for cost calculation")
            state["current_step"] = "currency_conversion"
            return state

        async def currency_conversion(state: dict) -> dict:
            if not state.get("currency"):
                raise ValueError("Currency not set")
            state["current_step"] = "generate_itinerary"
            return state

        async def generate_itinerary(state: dict) -> dict:
            if not all([state.get("location"), state.get("start_date"), state.get("end_date"), state.get("budget")]):
                raise ValueError("Missing required information for itinerary generation")
            state["travel_plan"] = await self.itinerary_service.generate_itinerary(
                state["location"],
                state["start_date"],
                state["end_date"],
                state["budget"],
                state["currency"]
            )
            state["current_step"] = "create_summary"
            return state

        def create_summary(state: dict) -> dict:
            if not state.get("travel_plan"):
                raise ValueError("Travel plan not generated")
            state["current_step"] = "complete"
            return state

        # Build the graph
        workflow = StateGraph(dict)
        # Add nodes
        workflow.add_node("init", init_state)
        workflow.add_node("search_attractions", search_attractions)
        workflow.add_node("search_weather", search_weather)
        workflow.add_node("search_hotels", search_hotels)
        workflow.add_node("calculate_costs", calculate_costs)
        workflow.add_node("currency_conversion", currency_conversion)
        workflow.add_node("generate_itinerary", generate_itinerary)
        workflow.add_node("create_summary", create_summary)
        # Add edges
        workflow.add_edge("init", "search_attractions")
        workflow.add_edge("search_attractions", "search_weather")
        workflow.add_edge("search_weather", "search_hotels")
        workflow.add_edge("search_hotels", "calculate_costs")
        workflow.add_edge("calculate_costs", "currency_conversion")
        workflow.add_edge("currency_conversion", "generate_itinerary")
        workflow.add_edge("generate_itinerary", "create_summary")
        # Set entry point
        workflow.set_entry_point("init")
        return workflow.compile()

    async def plan_trip(self, location: Location, start_date: datetime, 
                       end_date: datetime, budget: float, currency: str = "USD") -> TravelPlan:
        # Initialize the state as a dict
        state = {
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "budget": budget,
            "currency": currency,
            "travel_plan": None,
            "current_step": "init"
        }
        # Run the graph
        final_state = await self.graph.ainvoke(state)
        if not final_state.get("travel_plan"):
            raise ValueError("Failed to generate travel plan")
        return final_state["travel_plan"] 