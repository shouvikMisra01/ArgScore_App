import asyncio
import os
from main import analyze_argument, ArgumentRequest, client

# Mocking the request
async def run_test():
    print("Running debug test for 'Climate change is driven by human activities'...")
    req = ArgumentRequest(text="Climate change is significantly driven by human activities like burning fossil fuels.", numeric_tolerance=0.1)
    
    try:
        res = await analyze_argument(req)
        print("SUCCESS:", res)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test())
