import asyncio
from app.agent import app

async def main():
    print("Testing query: 'I have 2000 rupees to invest, is tatapowers a good option??'")
    
    # Run the workflow
    try:
        response = await app.run_async(
            "I have 2000 rupees to invest, is tatapowers a good option??"
        )
        print("\n=== AGENT RESPONSE ===")
        print(response)
        
        # In ADK 2.0, if response is a list or object, we can print its state/messages
        if hasattr(response, 'messages'):
            print("\n=== FINAL MESSAGE ===")
            print(response.messages[-1].content)
    except Exception as e:
        print(f"Workflow failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
