# Multi-Agent Investment Committee

An experiment in multi-agent AI: ten specialised agents (for example a risk officer and a macroeconomist) look at a
stock and a consensus agent reaches a BUY, SELL or HOLD decision. Built with Google ADK, FastAPI and Gemini 2.5 Flash.

- Agents must use hard numbers from Python tools (for example a Welch t-test, Altman Z-Score, Piotroski F-Score,
  Beneish M-Score from SEC filings) instead of free-text opinion
- Handles US tickers and Indian tickers (NSE/BSE, with Nifty 50 and RBI data)
- Guards: schema fallbacks so a bad data day does not crash the pipeline, a prompt-injection check, and a filter for personal data
- Run locally with `agents-cli playground`

A learning project, not financial advice.
