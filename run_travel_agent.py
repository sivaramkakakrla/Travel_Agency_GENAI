import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from models import Location
from agent import TravelAgent

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = ['WEATHER_API_KEY', 'HOTEL_API_KEY', 'ATTRACTION_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease set these variables in your .env file")
        return False
    return True

async def main():
    # Load environment variables
    load_dotenv()
    
    # Check environment setup
    if not check_environment():
        return

    try:
        # Create a travel agent
        print("Initializing travel agent...")
        agent = TravelAgent()
        print("\nGenerating travel plan...0000")
        # Define trip parameters
        location = Location(
            city="New Delhi",
            country="India",
            #coordinates={"lat": 48.8566, "lon": 2.3522}
        )
        print("\nGenerating travel plan...11111")
        start_date = datetime.now() + timedelta(days=30)  # Trip starts in 30 days
        end_date = start_date + timedelta(days=7)  # 7-day trip
        budget = 5000  # $5000 budget
        currency = "USD"

        print("\nTrip Parameters:")
        print(f"Destination: {location.city}, {location.country}")
        print(f"Dates: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Budget: {budget} {currency}")

        print("\nGenerating travel plan...")
        # Generate travel plan
        travel_plan = await agent.plan_trip(
            location=location,
            start_date=start_date,
            end_date=end_date,
            budget=budget,
            currency=currency
        )

        # Print the summary
        print("\n=== Travel Plan Summary ===")
        print(travel_plan.summary)
        
        print("\n=== Weather Forecast ===")
        print(f"Current temperature: {travel_plan.weather_forecast.temperature}°C")
        print(f"Condition: {travel_plan.weather_forecast.condition}")
        
        print("\n=== Budget Breakdown ===")
        for category, amount in travel_plan.budget.expenses.items():
            print(f"{category}: {amount} {currency}")
        
        print("\n=== Daily Itinerary ===")
        for day_plan in travel_plan.itinerary.daily_plans:
            print(f"\nDay: {day_plan.date.strftime('%Y-%m-%d')}")
            print("Activities:")
            for activity in day_plan.activities:
                print(f"- {activity.name} ({activity.duration})")
            print("Meals:")
            for meal in day_plan.meals:
                print(f"- {meal.name} ({meal.cuisine})")

    except ImportError as e:
        print(f"Error: Could not import required modules. Make sure all dependencies are installed.")
        print(f"Details: {str(e)}")
    except Exception as e:
        print(f"Error generating travel plan: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if all required API keys are set in .env file")
        print("2. Verify your internet connection")
        print("3. Make sure all dependencies are installed (pip install -r requirements.txt)")
        print("4. Check if the travel_agent package is properly installed")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {str(e)}") 