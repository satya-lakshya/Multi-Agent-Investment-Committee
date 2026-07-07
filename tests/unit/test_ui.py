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

import gradio as gr

from app.ui import demo, run_debug_tool


def test_ui_initialization() -> None:
    """Verifies that the Gradio Blocks demo object is constructed correctly."""
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "Institutional Investment Committee Dashboard"


def test_run_debug_tool_fallback() -> None:
    """Verifies that run_debug_tool catches exceptions and invalid names gracefully."""
    # Test invalid tool name
    res = run_debug_tool("invalid_tool_name", "{}")
    assert res["status"] == "error"
    assert "Unknown tool" in res["message"]

    # Test invalid json arguments
    res = run_debug_tool("test_pairs_cointegration", "{invalid_json}")
    assert res["status"] == "error"
    assert "Execution failed" in res["message"]
