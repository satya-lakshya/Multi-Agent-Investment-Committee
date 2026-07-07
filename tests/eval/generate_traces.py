# ruff: noqa: E402
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

import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Load .env first so GOOGLE_GENAI_USE_VERTEXAI=true (Vertex AI Premium) is active
# before any google.adk or google.genai imports happen
load_dotenv(override=True)

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from google.adk.events.request_input import RequestInput
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agent import app as adk_app


def serialize_event(ev):
    """Converts an ADK event object to a clean JSON-serializable dictionary."""
    res = {"author": ev.author or "assistant", "content": None}
    if ev.content:
        parts_list = []
        for part in ev.content.parts or []:
            part_dict = {}
            if part.text is not None:
                part_dict["text"] = part.text
            if part.function_call is not None:
                part_dict["function_call"] = {
                    "name": part.function_call.name,
                    "args": part.function_call.args,
                }
            if part.function_response is not None:
                part_dict["function_response"] = {
                    "name": part.function_response.name,
                    "response": part.function_response.response,
                }
            parts_list.append(part_dict)
        res["content"] = {"role": ev.content.role or "model", "parts": parts_list}
    return res


async def generate_traces():
    dataset_path = "tests/eval/datasets/basic-dataset.json"
    output_dir = "artifacts/traces"
    output_path = os.path.join(output_dir, "generated_traces.json")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)

    runner = InMemoryRunner(app=adk_app)
    eval_cases = []

    for idx, case in enumerate(dataset["eval_cases"]):
        case_id = case["eval_case_id"]
        prompt = case["prompt"]["parts"][0]["text"]
        print(f"\n--- Running Scenario: {case_id} ---")
        print(f"Prompt: {prompt}")

        # Preemptive sleep to respect Gemini free-tier rate limits
        if idx > 0:
            print("Sleeping 20s to respect free tier rate limit...")
            await asyncio.sleep(20)

        user_id = "eval_runner"

        max_attempts = 5
        attempt = 0
        success = False

        while attempt < max_attempts and not success:
            attempt += 1
            # Generate a unique session ID for each attempt to avoid AlreadyExistsError
            session_id = f"eval_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{attempt}"
            try:
                if attempt > 1:
                    print(f"Attempt {attempt}/{max_attempts}...")

                session = await runner.session_service.create_session(
                    app_name=adk_app.name, session_id=session_id, user_id=user_id
                )

                # Start execution
                is_running = True
                new_message = types.Content(
                    role="user", parts=[types.Part.from_text(text=prompt)]
                )

                while is_running:
                    is_running = False

                    async for event in runner.run_async(
                        user_id=user_id,
                        session_id=session.id,
                        new_message=new_message,
                    ):
                        new_message = None  # Clear new_message for next yields

                        # Check for RequestInput
                        if isinstance(event, RequestInput):
                            is_running = True
                            interrupt_id = event.interrupt_id

                            # Automate decision
                            if interrupt_id == "get_sector":
                                sector = "Information Technology"
                                if "biot" in prompt.lower():
                                    sector = "Biotech"
                                print(
                                    f"  [HITL] Intercepted get_sector. Responding with '{sector}'."
                                )
                                new_message = types.Content(
                                    role="user",
                                    parts=[
                                        types.Part(
                                            function_response=types.FunctionResponse(
                                                name="get_sector",
                                                id="get_sector",
                                                response={"get_sector": sector},
                                            )
                                        )
                                    ],
                                )
                            elif interrupt_id == "human_override":
                                decision = (
                                    "reject"
                                    if "override" in prompt.lower()
                                    or "injection" in prompt.lower()
                                    or "card" in prompt.lower()
                                    else "approve"
                                )
                                print(
                                    f"  [HITL] Intercepted human_override. Responding with '{decision}'."
                                )
                                new_message = types.Content(
                                    role="user",
                                    parts=[
                                        types.Part(
                                            function_response=types.FunctionResponse(
                                                name="human_override",
                                                id="human_override",
                                                response={"human_override": decision},
                                            )
                                        )
                                    ],
                                )
                            break
                success = True
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print("  [429 Quota Exceeded] Sleeping 55s before retrying...")
                    await asyncio.sleep(55)
                else:
                    print(f"  [Error] Execution failed: {e}")
                    raise e

        # Execution complete. Retrieve full session history
        final_session = await runner.session_service.get_session(
            app_name=adk_app.name, session_id=session.id, user_id=user_id
        )

        serialized_events = [serialize_event(ev) for ev in final_session.events]

        # Find the final text response
        final_response = ""
        for ev in reversed(final_session.events):
            if (
                ev.node_info
                and ev.node_info.name == "finish_node"
                and ev.content
                and ev.content.parts
            ):
                for p in ev.content.parts:
                    if p.text:
                        final_response = p.text
                        break
            if final_response:
                break

        if not final_response:
            for ev in reversed(final_session.events):
                if ev.content and ev.content.parts:
                    for p in ev.content.parts:
                        if p.text:
                            final_response = p.text
                            break
                if final_response:
                    break

        print(f"Completed {case_id}. Final Response Length: {len(final_response)}")

        eval_cases.append(
            {
                "eval_case_id": case_id,
                "prompt": case["prompt"],
                "responses": [
                    {"response": {"role": "model", "parts": [{"text": final_response}]}}
                ],
                "agent_data": {
                    "turns": [{"turn_index": 0, "events": serialized_events}]
                },
            }
        )

    print(f"\nWriting traces to {output_path}...")
    with open(output_path, "w") as f:
        json.dump({"eval_cases": eval_cases}, f, indent=2)
    print("Done!")


if __name__ == "__main__":
    asyncio.run(generate_traces())
