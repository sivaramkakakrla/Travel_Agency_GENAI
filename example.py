from datetime import datetime
from models import Location
from agent import TravelAgent

async def main():
    # Create a location
    location = Location(
        city="New York",
        country="United States",
        coordinates={'lat': 40.7128, 'lon': -74.0060}
    )
    
    # Create travel agent
    agent = TravelAgent()
    
    # Plan trip
    plan = await agent.plan_trip(
        location=location,
        start_date=datetime(2024, 6, 1),
        end_date=datetime(2024, 6, 5),
        budget=2000.0
    )
    
    print(plan.summary)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 