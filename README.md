# Autonomous Multi-Agent Investment Committee

An autonomous, multi-agent AI system simulating an institutional hedge fund committee. Engineered using Google ADK (Agent Development Kit), FastAPI, and Gemini 2.5 Flash, this system orchestrates 10 specialized AI agents via a graph-based workflow to evaluate global equities and reach a consensus-driven investment decision (BUY/SELL/HOLD).

## 🚀 Key Features

- **Complex Multi-Agent Orchestration:** Deploys strict personas (e.g., Chief Risk Officer, Lead Macroeconomist) with specific domain tools. A `consensus_evaluator` agent debates and synthesizes these distinct signals.
- **Algorithmic Rigor:** Bridges LLM reasoning with deterministic Python financial tools. Agents are forced to use hard quantitative metrics (e.g., rejecting signals if Welch’s t-test p-value is > 0.05).
- **Forensic Accounting Integration:** Automatically pulls SEC filings to calculate institutional-grade metrics:
  - Altman Z-Score (bankruptcy risk)
  - Piotroski F-Score (operational health)
  - Beneish M-Score (earnings manipulation)
- **Cross-Market Capability:** Intelligently detects user intent and automatically normalizes tickers and benchmarks for both the US Market (S&P 500 / Fed data) and the Indian Market (NSE/BSE / Nifty 50 / RBI data).

## 🛠️ Production-Ready Resilience

When transitioning AI agents from prototypes to production, pipelines often crash due to bad data or LLM hallucinations. This project implements several safeguards:

- **Deterministic Schema Fallbacks (Crash Prevention):** Implemented strict Pydantic output schemas combined with a "Critical Schema Compliance Rule" in agent prompts. The system safely falls back to explicit default values rather than dropping JSON keys during total data failures (e.g., when querying micro-caps).
- **Cross-Market Data Normalization & Routing:** A context-aware normalization engine auto-detects Indian market intent, appends `.NS`/`.BO` suffixes, and dynamically routes comparative analysis to the correct regional benchmarks (Nifty 50 vs. SPY).
- **Safe Serialization Wrapper:** Developed a highly resilient serialization guardrail that safely converts mixed object types, JSON strings, and Pydantic models into valid dictionaries to prevent mid-workflow failures.

## 🔒 Security Measures

- **Prompt Injection & LLM Override Prevention:** An upstream `is_security_violation` interceptor scans raw user queries before they reach the LLM, blocking adversarial prompt injection attempts (e.g., "ignore previous instructions").
- **PII Leakage Prevention:** Integrated Regex-based sanitization algorithms to scan and block requests containing sensitive user data (Emails, Phone Numbers, SSNs).

## 💻 Tech Stack

- **Frameworks:** Google ADK, FastAPI
- **AI/LLM:** Gemini 2.5 Flash
- **Data & Finance Tools:** Pandas, Statsmodels, yfinance
- **Infrastructure:** Docker, Google Cloud (Trace, BigQuery)

## 🏃‍♂️ Quick Start

Install dependencies using `uv`:

```bash
uvx google-agents-cli setup
agents-cli install
```

Launch the local development environment:

```bash
agents-cli playground
```

## 🚀 Deployment

This agent is configured to deploy to Google Cloud Agent Runtime.

```bash
gcloud config set project <your-project-id>
agents-cli deploy
```
