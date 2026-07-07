import logging
import sys
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types
from app.agent import root_agent

# Configure root logger to output to stdout
root = logging.getLogger()
root.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def main():
    query = "i have 2000 rupees to invest, is tatapower a good option?"
    print(f"Running workflow for query: '{query}'", flush=True)
    
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    message = types.Content(
        role="user", parts=[types.Part.from_text(text=query)]
    )

    try:
        # Avoid list() to stream events in real-time
        for event in runner.run(
            new_message=message,
            user_id="test_user",
            session_id=session.id,
            run_config=RunConfig(streaming_mode=StreamingMode.SSE),
        ):
            if hasattr(event, "output") and event.output:
                print(f"EVENT OUTPUT: {type(event.output)} -> {event.output}", flush=True)
            elif event.content and event.content.parts:
                print(f"EVENT CONTENT: {event.content.parts[0].text}", flush=True)
            else:
                print(f"EVENT: {event}", flush=True)
        print("Workflow completed successfully.", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
