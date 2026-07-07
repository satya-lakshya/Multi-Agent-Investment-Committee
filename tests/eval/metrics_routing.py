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

from google import genai
from google.genai import types
from pydantic import BaseModel


class _Verdict(BaseModel):
    score: int  # 1-5
    explanation: str


def evaluate(instance):
    """LLM-as-judge evaluating routing and debate correctness."""
    prompt = (
        "You are an expert QA evaluator for an enterprise multi-agent investment workflow.\n"
        "Your task is to judge the routing and debate correctness of the workflow trace on a scale of 1 to 5.\n"
        "\n"
        "Evaluation Criteria:\n"
        "1. Discrepancy Detection & Routing: If there is a significant discrepancy between worker agents "
        "(e.g., Valuation Expert is bullish/undervalued but Risk Analyst/CRO rated the overall risk as HIGH or EXTREME, "
        "or Portfolio Manager proposed BUY but Risk is HIGH/EXTREME), the debate node MUST be triggered.\n"
        "2. Debate Turn Budget Containment: The debate must proceed under a strict hard limit of exactly 1 debate turn. "
        "The debate_turn_count must never exceed 1.\n"
        "\n"
        "Scoring Rubric:\n"
        "- Score 1: Severe failure. A discrepancy occurred but no debate was triggered, OR the debate ran for more than 1 turn.\n"
        "- Score 2: Major issue. Debate triggered but exceeded the 1-turn budget.\n"
        "- Score 3: Moderate issue. Debate triggered and run but resolution/routing logic was confused or incomplete.\n"
        "- Score 4: Good. Debate triggered appropriately and concluded within exactly 1 turn.\n"
        "- Score 5: Excellent. Discrepancies were identified, debate triggered, and resolved with professional reasoning in exactly 1 debate turn.\n"
        "\n"
        f"User Prompt: {instance.get('prompt', '')}\n"
        f"Final Response: {instance.get('response', '')}\n"
        f"Full Agent Trace:\n{instance.get('agent_data', '')}\n"
    )

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,  # deterministic grading
            response_mime_type="application/json",
            response_schema=_Verdict,
        ),
    )

    verdict = response.parsed
    if verdict is None:
        return {
            "score": 0,
            "explanation": response.text or "Model returned no JSON verdict.",
        }
    return {"score": max(1, min(5, verdict.score)), "explanation": verdict.explanation}
