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
import logging
import os
from datetime import datetime

import gradio as gr
from google.adk.runners import InMemoryRunner
from google.genai import types

from app.agent import app as adk_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ui-playground")

# Initialize Runner
runner = InMemoryRunner(app=adk_app)

# Dictionary to hold session-based debugger events
debug_events: dict[str, asyncio.Event] = {}

# Custom CSS for dark-mode glassmorphism and modern premium styling
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: radial-gradient(circle at 50% 0%, #0b0f19, #020408) !important;
    color: #f3f4f6 !important;
}

/* Premium Card Layouts */
.agent-card {
    background: rgba(13, 20, 35, 0.45) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
}

.agent-card:hover {
    transform: translateY(-2px) !important;
    border-color: rgba(59, 130, 246, 0.3) !important;
    box-shadow: 0 10px 30px -10px rgba(59, 130, 246, 0.15) !important;
}

/* Specific Role Borders */
.quant-card { border-left: 4px solid #3b82f6 !important; }
.financial-card { border-left: 4px solid #10b981 !important; }
.valuation-card { border-left: 4px solid #8b5cf6 !important; }
.macro-card { border-left: 4px solid #f59e0b !important; }
.sentiment-card { border-left: 4px solid #ec4899 !important; }
.risk-card { border-left: 4px solid #ef4444 !important; }
.institutional-card { border-left: 4px solid #06b6d4 !important; }
.pm-card { border-left: 4px solid #6366f1 !important; }

/* Status Rings & Badges */
.status-label {
    font-weight: 700 !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
    display: inline-block !important;
}

.status-idle { color: #6b7280 !important; background: rgba(107, 114, 128, 0.1) !important; }
.status-running { color: #f59e0b !important; background: rgba(245, 158, 11, 0.1) !important; animation: pulse 2s infinite !important; }
.status-completed { color: #10b981 !important; background: rgba(16, 185, 129, 0.1) !important; }
.status-debating { color: #a78bfa !important; background: rgba(167, 139, 250, 0.1) !important; animation: pulse 1.5s infinite !important; }
.status-paused { color: #3b82f6 !important; background: rgba(59, 130, 246, 0.15) !important; border: 1px dashed #3b82f6 !important; }

@keyframes pulse {
    0% { opacity: 0.6; }
    50% { opacity: 1; }
    100% { opacity: 0.6; }
}

/* Console logs & inspector styles */
.terminal-box textarea {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #03050a !important;
    border: 1px solid #111827 !important;
    color: #38bdf8 !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    border-radius: 8px !important;
}

.report-markdown {
    font-family: 'Outfit', sans-serif !important;
    line-height: 1.7 !important;
}

.report-markdown h1, .report-markdown h2, .report-markdown h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}
"""


def generate_graph_html(active_node: str, states: dict[str, str]) -> str:
    """Generates a premium SVG layout representing the multi-agent graph with glowing status states."""

    def get_color(node_name: str) -> str:
        status = states.get(node_name, "IDLE")
        if status == "RUNNING":
            return "#f59e0b"  # Amber
        if status == "COMPLETED":
            return "#10b981"  # Emerald Green
        if status == "DEBATING":
            return "#8b5cf6"  # Purple
        if status == "PAUSED":
            return "#3b82f6"  # Blue
        return "#1f2937"  # Dark Grey

    def get_glow(node_name: str) -> str:
        status = states.get(node_name, "IDLE")
        if status == "RUNNING":
            return "filter='url(#glow-running)'"
        if status == "COMPLETED":
            return "filter='url(#glow-completed)'"
        if status == "DEBATING":
            return "filter='url(#glow-debating)'"
        if status == "PAUSED":
            return "filter='url(#glow-paused)'"
        return ""

    # Generate node status indicator dots
    svg_nodes = {
        "START": {"x": 40, "y": 180, "label": "START"},
        "PM": {"x": 140, "y": 180, "label": "PM (CIO)"},
        "QUANT": {"x": 300, "y": 30, "label": "Quant Analyst"},
        "FINANCIAL": {"x": 300, "y": 80, "label": "Financial Forensic"},
        "VALUATION": {"x": 300, "y": 130, "label": "Valuation Expert"},
        "MACRO": {"x": 300, "y": 180, "label": "Macro Strategist"},
        "SENTIMENT": {"x": 300, "y": 230, "label": "Sentiment NLP"},
        "RISK": {"x": 300, "y": 280, "label": "Chief Risk Officer"},
        "INSTITUTIONAL": {"x": 300, "y": 330, "label": "Institutional flow"},
        "JOIN": {"x": 460, "y": 180, "label": "Join Node"},
        "DEBATE": {"x": 460, "y": 60, "label": "Debate Node"},
        "HUMAN": {"x": 140, "y": 300, "label": "Security Override"},
        "FINISH": {"x": 590, "y": 180, "label": "Report Output"},
    }

    # Generate connecting lines
    lines_html = ""
    # Start -> PM
    lines_html += "<path d='M 80 180 L 110 180' stroke='#374151' stroke-width='2' marker-end='url(#arrow)' />"

    # PM -> Parallel Analysts
    for k in [
        "QUANT",
        "FINANCIAL",
        "VALUATION",
        "MACRO",
        "SENTIMENT",
        "RISK",
        "INSTITUTIONAL",
    ]:
        ny = svg_nodes[k]["y"]
        lines_html += f"<path d='M 180 180 C 220 180, 220 {ny}, 260 {ny}' stroke='#374151' stroke-width='2' fill='none' marker-end='url(#arrow)' />"

    # Parallel Analysts -> Join Node
    for k in [
        "QUANT",
        "FINANCIAL",
        "VALUATION",
        "MACRO",
        "SENTIMENT",
        "RISK",
        "INSTITUTIONAL",
    ]:
        ny = svg_nodes[k]["y"]
        lines_html += f"<path d='M 410 {ny} C 430 {ny}, 430 180, 450 180' stroke='#374151' stroke-width='2' fill='none' />"

    # Join Node -> PM
    lines_html += "<path d='M 470 195 L 470 380 L 140 380 L 140 205' stroke='#374151' stroke-dasharray='4' stroke-width='2' fill='none' marker-end='url(#arrow)' />"

    # PM -> Human review
    lines_html += "<path d='M 140 205 L 140 270' stroke='#374151' stroke-width='2' marker-end='url(#arrow)' />"

    # PM -> Debate
    lines_html += "<path d='M 160 170 C 180 120, 360 60, 420 60' stroke='#374151' stroke-width='2' fill='none' marker-end='url(#arrow)' />"

    # Debate -> PM
    lines_html += "<path d='M 460 75 C 430 100, 220 160, 170 175' stroke='#374151' stroke-dasharray='4' stroke-width='2' fill='none' marker-end='url(#arrow)' />"

    # PM -> Finish
    lines_html += "<path d='M 160 180 C 260 180, 500 180, 560 180' stroke='#374151' stroke-width='2' fill='none' marker-end='url(#arrow)' />"

    # Draw nodes
    nodes_html = ""
    for name, coords in svg_nodes.items():
        color = get_color(name)
        glow = get_glow(name)
        active_border = (
            "stroke='#60a5fa' stroke-width='2'"
            if active_node == name
            else "stroke='#374151' stroke-width='1'"
        )

        # Round rect for node label
        nodes_html += f"""
        <g id="node-{name}">
            <rect x="{coords["x"] - 30}" y="{coords["y"] - 15}" width="110" height="30" rx="8" fill="{color}" {active_border} {glow} />
            <text x="{coords["x"] + 25}" y="{coords["y"] + 5}" fill="#ffffff" font-size="9" font-family="'Outfit', sans-serif" font-weight="600" text-anchor="middle">{coords["label"]}</text>
        </g>
        """

    # Build full SVG
    svg_html = f"""
    <svg width="100%" height="240" viewBox="0 0 700 390" fill="none" xmlns="http://www.w3.org/2000/svg" style="background: rgba(13, 20, 35, 0.4); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); padding: 10px;">
        <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 1 L 10 5 L 0 9 z" fill="#4b5563" />
            </marker>

            <filter id="glow-running" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="6" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-completed" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="6" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-debating" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="6" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
            <filter id="glow-paused" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="6" result="blur" />
                <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
        </defs>
        <style>
            rect {{ transition: fill 0.4s ease, filter 0.4s ease; }}
        </style>
        {lines_html}
        {nodes_html}
    </svg>
    """
    return svg_html


async def run_workflow_stream(ticker, capital, horizon, risk_params, enable_debugger):
    # Create session
    session_id = f"ui_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    user_id = "ui_user"

    # Initialize debugger event for this session
    debug_event = asyncio.Event()
    debug_events[session_id] = debug_event

    # Initial state values mapping
    state_dict = {}

    # Track node statuses locally for visualization
    node_states = {
        "START": "COMPLETED",
        "PM": "RUNNING",
        "QUANT": "IDLE",
        "FINANCIAL": "IDLE",
        "VALUATION": "IDLE",
        "MACRO": "IDLE",
        "SENTIMENT": "IDLE",
        "RISK": "IDLE",
        "INSTITUTIONAL": "IDLE",
        "JOIN": "IDLE",
        "DEBATE": "IDLE",
        "HUMAN": "IDLE",
        "FINISH": "IDLE",
    }

    # Clear and set output defaults
    for name in [
        "quant",
        "financial",
        "valuation",
        "macro",
        "sentiment",
        "risk",
        "institutional",
        "pm",
    ]:
        state_dict[status_outputs[name]] = "IDLE"
        state_dict[data_outputs[name]] = {}

    state_dict[terminal_log] = "Initializing Multi-Agent Investment Workflow...\n"
    state_dict[debate_count] = 0
    state_dict[decision_gauge] = "IDLE"
    state_dict[final_markdown] = ""
    state_dict[state_json_viewer] = {}
    state_dict[graph_visualizer] = generate_graph_html("PM", node_states)
    state_dict[error_box] = gr.update(visible=False)
    yield state_dict

    query = f"Analyze {ticker.strip()} in the Technology sector. Capital: ${capital:,}, Horizon: {horizon}, Constraints: {risk_params}"
    logs = f"User Query: {query}\n"

    try:
        session = await runner.session_service.create_session(
            app_name=adk_app.name, session_id=session_id, user_id=user_id
        )

        # Keep track of debugger pause flags to avoid multiple pauses
        paused_proposal = False
        paused_analysts = False
        paused_valuation = False

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=types.Content(
                role="user", parts=[types.Part.from_text(text=query)]
            ),
        ):
            # Refresh live session state
            current_session = await runner.session_service.get_session(
                app_name=adk_app.name, session_id=session.id, user_id=user_id
            )
            state = current_session.state if current_session else {}
            state_dict[state_json_viewer] = state

            # Map events to logs
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        logs += f"[Agent Event] {part.text.strip()}\n"
                    elif part.function_call:
                        logs += (
                            f"[Deterministic Tool] Running: {part.function_call.name}\n"
                        )

            if event.output is not None:
                logs += f"[Node Complete] Output: {str(event.output)[:120]}...\n"

            # Parse Active Node Path for SVG highlighting
            active_node = "START"
            if event.node_info and event.node_info.path:
                active_node = event.node_info.path.split("/")[-1].split("@")[0].upper()

            # Synchronize node states from ADK session variables
            for key, name in [
                ("quant_analysis", "QUANT"),
                ("financial_analysis", "FINANCIAL"),
                ("valuation_analysis", "VALUATION"),
                ("macro_analysis", "MACRO"),
                ("sentiment_analysis", "SENTIMENT"),
                ("risk_analysis", "RISK"),
                ("institutional_analysis", "INSTITUTIONAL"),
                ("consensus", "FINISH"),
            ]:
                if state.get(key):
                    node_states[name] = "COMPLETED"
                elif active_node == name:
                    node_states[name] = "RUNNING"

            # Special Node Highlights
            if state.get("proposal"):
                node_states["PM"] = "COMPLETED"
            if active_node == "PORTFOLIO_MANAGER":
                node_states["PM"] = "RUNNING"
            if state.get("debate_triggered"):
                node_states["DEBATE"] = "COMPLETED"
                node_states["VALUATION"] = "DEBATING"
            if active_node == "DEBATE_NODE":
                node_states["DEBATE"] = "RUNNING"
            if active_node == "HUMAN_REVIEW":
                node_states["HUMAN"] = "RUNNING"

            # Map values to Gradio UI cards
            state_dict[terminal_log] = logs
            state_dict[debate_count] = state.get("debate_turn_count", 0)
            state_dict[graph_visualizer] = generate_graph_html(active_node, node_states)

            for key, name in [
                ("quant_analysis", "quant"),
                ("financial_analysis", "financial"),
                ("valuation_analysis", "valuation"),
                ("macro_analysis", "macro"),
                ("sentiment_analysis", "sentiment"),
                ("risk_analysis", "risk"),
                ("institutional_analysis", "institutional"),
                ("consensus", "pm"),
            ]:
                payload = state.get(key)
                if payload:
                    if name == "valuation" and state.get("debate_triggered"):
                        state_dict[status_outputs[name]] = "COMPLETED (Debated)"
                        state_dict[data_outputs[name]] = (
                            state.get("revised_valuation") or payload
                        )
                    else:
                        state_dict[status_outputs[name]] = "COMPLETED"
                        state_dict[data_outputs[name]] = payload
                else:
                    if (
                        name in ("valuation", "risk")
                        and state.get("debate_triggered")
                        and not state.get("revised_valuation")
                    ):
                        state_dict[status_outputs[name]] = "DEBATING"
                    else:
                        state_dict[status_outputs[name]] = (
                            "RUNNING" if active_node == name.upper() else "IDLE"
                        )

            # If final decision is reached
            consensus = state.get("consensus")
            if consensus:
                state_dict[decision_gauge] = consensus.get("final_decision", "HOLD")

            yield state_dict

            # -----------------------------------------------------------------
            # Interactive Debugger Pause Gates
            # -----------------------------------------------------------------
            if enable_debugger:
                # Gate 1: Pause after Initial Proposal is formed
                if state.get("proposal") and not paused_proposal:
                    paused_proposal = True
                    node_states["PM"] = "PAUSED"
                    state_dict[graph_visualizer] = generate_graph_html(
                        "PM", node_states
                    )
                    logs += "\n[DEBUGGER] Paused at Gate 1: Initial Proposal Generated.\nInspect the state and click 'Step Forward' to launch parallel analysts.\n"
                    state_dict[terminal_log] = logs
                    yield state_dict
                    await debug_event.wait()
                    debug_event.clear()
                    node_states["PM"] = "COMPLETED"
                    logs += "[DEBUGGER] Resuming execution...\n"
                    state_dict[terminal_log] = logs
                    yield state_dict

                # Gate 2: Pause after parallel execution is completed (join node)
                analysts_done = all(
                    state.get(k) is not None
                    for k in [
                        "quant_analysis",
                        "financial_analysis",
                        "valuation_analysis",
                        "macro_analysis",
                        "sentiment_analysis",
                        "risk_analysis",
                        "institutional_analysis",
                    ]
                )
                if analysts_done and not paused_analysts:
                    paused_analysts = True
                    node_states["JOIN"] = "PAUSED"
                    state_dict[graph_visualizer] = generate_graph_html(
                        "JOIN", node_states
                    )
                    logs += "\n[DEBUGGER] Paused at Gate 2: Parallel Analyst Analysis Complete.\nAll metrics gathered. Click 'Step Forward' to evaluate consensus/triggers.\n"
                    state_dict[terminal_log] = logs
                    yield state_dict
                    await debug_event.wait()
                    debug_event.clear()
                    node_states["JOIN"] = "COMPLETED"
                    logs += "[DEBUGGER] Resuming execution...\n"
                    state_dict[terminal_log] = logs
                    yield state_dict

                # Gate 3: Pause after debate node runs (before final consensus check)
                if (
                    state.get("debate_triggered")
                    and state.get("revised_valuation")
                    and not paused_valuation
                ):
                    paused_valuation = True
                    node_states["DEBATE"] = "PAUSED"
                    state_dict[graph_visualizer] = generate_graph_html(
                        "DEBATE", node_states
                    )
                    logs += "\n[DEBUGGER] Paused at Gate 3: Discrepancy Debate Complete.\nRevised valuation is locked. Click 'Step Forward' to run final consensus.\n"
                    state_dict[terminal_log] = logs
                    yield state_dict
                    await debug_event.wait()
                    debug_event.clear()
                    node_states["DEBATE"] = "COMPLETED"
                    logs += "[DEBUGGER] Resuming execution...\n"
                    state_dict[terminal_log] = logs
                    yield state_dict

        # Final layout updates
        final_session = await runner.session_service.get_session(
            app_name=adk_app.name, session_id=session.id, user_id=user_id
        )
        final_state = final_session.state if final_session else {}

        report_md = ""
        # Find markdown output in event history or fallback
        for event in final_session.events if final_session else []:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text and "#" in part.text:
                        report_md = part.text

        if not report_md:
            consensus_check = final_state.get("consensus", {})
            report_md = f"# 🏛️ Institutional Investment Committee Report\n\n## Decision: {consensus_check.get('final_decision')}\n\n{consensus_check.get('rationale')}"

        state_dict[final_markdown] = report_md
        state_dict[decision_gauge] = final_state.get("consensus", {}).get(
            "final_decision", "HOLD"
        )

        # Highlight final output node in green
        for n in node_states:
            node_states[n] = "COMPLETED"
        state_dict[graph_visualizer] = generate_graph_html("FINISH", node_states)
        yield state_dict

    except Exception as e:
        logger.exception("UI Workflow execution failed")
        state_dict[error_box] = gr.update(value=f"Workflow error: {e!s}", visible=True)
        state_dict[terminal_log] = (
            logs + f"\n[CRITICAL ERROR] Execution failed: {e!s}\n"
        )
        yield state_dict
    finally:
        # Clean up events
        debug_events.pop(session_id, None)


async def trigger_step(session_viewer):
    """Signals the debugger event to step forward."""
    # Retrieve current active session ID from state viewer
    session_id = session_viewer.get("session_id")
    if not session_id:
        # Get from timestamp if missing
        active_sessions = list(debug_events.keys())
        if active_sessions:
            session_id = active_sessions[-1]

    if session_id and session_id in debug_events:
        logger.info(f"Triggering step forward for session: {session_id}")
        debug_events[session_id].set()
        return "Resuming execution path..."
    return "No active debug session found."


def run_debug_tool(tool_name, args_json):
    from app.institutional_tools import (
        analyze_block_trade_volume,
        fetch_insider_transactions,
        fetch_institutional_holdings,
    )
    from app.quant_tools import (
        calculate_factor_information_coefficient,
        test_pairs_cointegration,
        validate_signal_statistics,
    )
    from app.risk_tools import (
        analyze_debt_maturity_wall,
        calculate_tail_risk_metrics,
        extract_concentration_risks,
        run_monte_carlo_stress_test,
    )
    from app.valuation_tools import (
        calculate_absolute_dcf,
        calculate_graham_intrinsic_value,
        calculate_margin_of_safety,
        calculate_relative_multiples,
    )

    tools_map = {
        "test_pairs_cointegration": test_pairs_cointegration,
        "calculate_factor_information_coefficient": calculate_factor_information_coefficient,
        "validate_signal_statistics": validate_signal_statistics,
        "calculate_tail_risk_metrics": calculate_tail_risk_metrics,
        "run_monte_carlo_stress_test": run_monte_carlo_stress_test,
        "analyze_debt_maturity_wall": analyze_debt_maturity_wall,
        "extract_concentration_risks": extract_concentration_risks,
        "fetch_insider_transactions": fetch_insider_transactions,
        "fetch_institutional_holdings": fetch_institutional_holdings,
        "analyze_block_trade_volume": analyze_block_trade_volume,
        "calculate_absolute_dcf": calculate_absolute_dcf,
        "calculate_graham_intrinsic_value": calculate_graham_intrinsic_value,
        "calculate_relative_multiples": calculate_relative_multiples,
        "calculate_margin_of_safety": calculate_margin_of_safety,
    }

    try:
        args = json.loads(args_json) if args_json.strip() else {}
        if tool_name not in tools_map:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        # Execute the tool
        res = tools_map[tool_name](**args)
        return res
    except Exception as e:
        return {"status": "error", "message": f"Execution failed: {e!s}"}


# Define Gradio blocks app
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
).set(
    body_background_fill="*neutral_950",
    block_background_fill="*neutral_900",
    block_border_color="*neutral_800",
)

with gr.Blocks(title="Institutional Investment Committee Dashboard") as demo:
    gr.HTML(
        "<h1 style='text-align: center; font-size: 2.2rem; font-weight: 800; background: linear-gradient(to right, #6366f1, #3b82f6, #10b981); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;'>🏛️ Institutional Investment Committee Dashboard</h1>"
    )
    gr.HTML(
        "<p style='text-align: center; color: #6b7280; margin-bottom: 1.5rem;'>ADK 2.0 Stateful Multi-Agent Looping Consensus Workflow Playground</p>"
    )

    error_box = gr.Textbox(label="Debugger Error Log", visible=False, interactive=False)

    with gr.Tabs():
        # TAB 1: COMMITTEE WORKSPACE
        with gr.TabItem("📊 Execution Workspace"):
            with gr.Row():
                # Sidebar controls
                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("### ⚙️ Parameters")
                    ticker_input = gr.Textbox(
                        value="INFY",
                        label="Ticker Symbol",
                        placeholder="e.g. INFY, AAPL, HDFCBANK",
                    )
                    capital_input = gr.Slider(
                        minimum=10000,
                        maximum=100000000,
                        value=5000000,
                        step=10000,
                        label="Capital Allocation ($)",
                    )
                    horizon_input = gr.Dropdown(
                        choices=["1 Year", "2 Years", "3 Years", "5 Years"],
                        value="3 Years",
                        label="Investment Horizon",
                    )
                    risk_input = gr.Textbox(
                        value="Minimizing beta, target Sharpe ratio > 1.5, protect downside tail risk",
                        label="Custom Risk Constraints",
                    )

                    gr.Markdown("---")
                    gr.Markdown("### 🛠️ Debugger controls")
                    enable_debugger = gr.Checkbox(
                        label="Enable Step-by-Step Debugger", value=True
                    )
                    step_btn = gr.Button("➡️ Step Forward", variant="secondary")

                    run_btn = gr.Button("🚀 Run Committee Workflow", variant="primary")

                    gr.Markdown("---")
                    debate_count = gr.Number(
                        value=0, label="Debate Turn Count", interactive=False
                    )
                    decision_gauge = gr.Textbox(
                        value="IDLE", label="Recommendation Decision", interactive=False
                    )

                # Workspace and Visualizer
                with gr.Column(scale=3):
                    gr.Markdown("### 🗺️ Real-time Workflow Execution Map")
                    graph_visualizer = gr.HTML(value=generate_graph_html("START", {}))

                    gr.Markdown("### 💬 Parallel Agent Live Outputs")

                    status_outputs = {}
                    data_outputs = {}

                    agents = [
                        ("quant", "Quantitative Analyst", "quant-card"),
                        ("financial", "Financial Analyst (Forensic)", "financial-card"),
                        ("valuation", "Valuation Expert", "valuation-card"),
                        ("macro", "Macro Strategist", "macro-card"),
                        ("sentiment", "Behavioral NLP (Sentiment)", "sentiment-card"),
                        ("risk", "Chief Risk Officer", "risk-card"),
                        (
                            "institutional",
                            "Order Flow (Smart Money)",
                            "institutional-card",
                        ),
                        ("pm", "Portfolio Manager (Consensus)", "pm-card"),
                    ]

                    # 4x2 Grid of cards
                    for r in range(2):
                        with gr.Row():
                            for c in range(4):
                                idx = r * 4 + c
                                agent_key, agent_name, card_class = agents[idx]

                                with gr.Group(elem_classes=["agent-card", card_class]):
                                    gr.Markdown(f"**{agent_name}**")
                                    status_outputs[agent_key] = gr.Textbox(
                                        value="IDLE",
                                        label="Phase",
                                        elem_classes=["status-label"],
                                        interactive=False,
                                    )
                                    data_outputs[agent_key] = gr.JSON(
                                        value={}, label="Payload JSON"
                                    )

        # TAB 2: EXECUTIVE REPORT
        with gr.TabItem("📝 Executive Briefing"):
            with gr.Row():
                with gr.Column(scale=3):
                    final_markdown = gr.Markdown(
                        value="*Run the workflow committee in the Workspace tab to generate the final report.*",
                        elem_classes=["report-markdown"],
                    )
                with gr.Column(scale=1):
                    terminal_log = gr.Textbox(
                        label="Real-time Execution & Debate Log",
                        lines=22,
                        max_lines=28,
                        elem_classes=["terminal-box"],
                        interactive=False,
                    )

        # TAB 3: STATE INSPECTOR
        with gr.TabItem("🔍 Live State Inspector"):
            gr.Markdown(
                "Inspect the raw ADK `WorkflowState` object variables in real time as the execution proceeds."
            )
            state_json_viewer = gr.JSON(value={}, label="Live WorkflowState JSON Tree")

        # TAB 4: TOOL SANDBOX
        with gr.TabItem("🛠️ Tool Sandbox"):
            gr.Markdown(
                "Directly execute and test individual deterministic Python analysis tools with custom JSON inputs."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    debug_tool_name = gr.Dropdown(
                        choices=[
                            "test_pairs_cointegration",
                            "calculate_factor_information_coefficient",
                            "validate_signal_statistics",
                            "calculate_tail_risk_metrics",
                            "run_monte_carlo_stress_test",
                            "analyze_debt_maturity_wall",
                            "extract_concentration_risks",
                            "fetch_insider_transactions",
                            "fetch_institutional_holdings",
                            "analyze_block_trade_volume",
                            "calculate_absolute_dcf",
                            "calculate_graham_intrinsic_value",
                            "calculate_relative_multiples",
                            "calculate_margin_of_safety",
                        ],
                        value="test_pairs_cointegration",
                        label="Select Analysis Tool",
                    )
                    debug_args = gr.Textbox(
                        value='{"ticker_a": "INFY", "ticker_b": "TCS", "historical_days": 252}',
                        label="JSON Arguments Payload",
                    )
                    debug_run_btn = gr.Button("Execute Tool", variant="secondary")
                with gr.Column(scale=2):
                    debug_output = gr.JSON(label="Tool Output Payload")

            debug_run_btn.click(
                fn=run_debug_tool,
                inputs=[debug_tool_name, debug_args],
                outputs=debug_output,
            )

    # Set up step event handlers
    step_btn.click(fn=trigger_step, inputs=[state_json_viewer], outputs=[error_box])

    # Set up button click handler
    all_outputs = (
        list(status_outputs.values())
        + list(data_outputs.values())
        + [
            terminal_log,
            debate_count,
            decision_gauge,
            final_markdown,
            error_box,
            state_json_viewer,
            graph_visualizer,
        ]
    )

    run_btn.click(
        fn=run_workflow_stream,
        inputs=[
            ticker_input,
            capital_input,
            horizon_input,
            risk_input,
            enable_debugger,
        ],
        outputs=all_outputs,
    )

if __name__ == "__main__":
    demo.queue()
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, css=CUSTOM_CSS, theme=theme)
