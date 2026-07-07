# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import BaseModel

from app.agent import app as adk_app

# 1. Logging Configuration
# Use standard Python logging for console logs as requested
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ambient-investment-committee")

# 2. Telemetry Configuration
# Explicitly set otel_to_cloud=False (implicitly off as we do not run via ADK CLI api-server)
os.environ["ADK_OTEL_TO_CLOUD"] = "False"

# Initialize FastAPI
app = FastAPI(title="Ambient Investment Committee Web Service")

# Initialize Runner
runner = InMemoryRunner(app=adk_app)


# Pub/Sub Payload Schemas
class PubSubMessage(BaseModel):
    data: str | None = None
    attributes: dict[str, str] | None = None
    messageId: str | None = None
    message_id: str | None = None
    publishTime: str | None = None
    publish_time: str | None = None


class PubSubPushRequest(BaseModel):
    message: PubSubMessage
    subscription: str


@app.post("/pubsub")
async def handle_pubsub_push(payload: PubSubPushRequest):
    """Handles incoming Pub/Sub push messages, normalizes subscriptions, and executes the agent."""
    subscription = payload.subscription
    message = payload.message

    # Normalize fully-qualified subscription path down to the short name
    # e.g., "projects/my-project/subscriptions/infosys-alert-sub" -> "infosys-alert-sub"
    normalized_session_id = subscription.split("/")[-1]
    logger.info(
        "Received Pub/Sub message. Subscription: %s (Normalized Session ID: %s)",
        subscription,
        normalized_session_id,
    )

    # Base64 decode data payload
    if not message.data:
        logger.warning("Empty data payload received in Pub/Sub message.")
        raise HTTPException(status_code=400, detail="Empty data payload.")

    try:
        decoded_bytes = base64.b64decode(message.data)
        decoded_text = decoded_bytes.decode("utf-8")
    except Exception as e:
        logger.error("Failed to base64 decode message data: %s", e)
        raise HTTPException(
            status_code=400, detail="Invalid base64 payload data."
        ) from e

    logger.info("Decoded payload data: %s", decoded_text)

    # Check if decoded payload is JSON (e.g. structured alert)
    try:
        json_data = json.loads(decoded_text)
        ticker = json_data.get("ticker", "Unknown")
        event_name = json_data.get("event", "price_move")
        details = json_data.get("details", "")
        query = f"Ambient market event triggered for {ticker}. Event: {event_name}. Details: {details or 'market movement'}"
    except (json.JSONDecodeError, TypeError):
        # Fall back to raw decoded text
        query = decoded_text

    logger.info("Formulated Query: '%s'", query)

    # Drive the workflow session in InMemoryRunner
    user_id = "ambient_system"
    session_id = normalized_session_id

    # Create/Get session
    session = await runner.session_service.get_session(
        app_name="app", session_id=session_id, user_id=user_id
    )
    if session is None:
        session = await runner.session_service.create_session(
            app_name="app", session_id=session_id, user_id=user_id
        )

    results = []
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text=query)]
            ),
        ):
            if event.output is not None:
                logger.info("Agent Output: %s", event.output)
                results.append(event.output)
            if event.content is not None and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        logger.info("Agent Event Content: %s", part.text)
                    elif part.function_call:
                        logger.warning(
                            "Agent requested input/interrupt: %s",
                            part.function_call.name,
                        )
    except Exception as e:
        logger.exception("Error executing ambient workflow: %s", e)
        raise HTTPException(
            status_code=500, detail=f"Workflow execution failed: {e}"
        ) from e

    return {
        "status": "success",
        "subscription": subscription,
        "session_id": session_id,
        "results": results,
    }


def main():
    # Clear API keys to ensure Vertex AI uses GCP Application Default Credentials
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]

    logger.info("Starting ambient web server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
