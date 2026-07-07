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

import os
import re
from typing import Any, Literal

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.models import Gemini
from google.adk.workflow import JoinNode, Workflow, node
from google.genai import types
from pydantic import BaseModel, Field

from app.financial_tools import (
    analyze_earnings_quality,
    compute_forensic_scores,
    evaluate_graham_criteria,
    fetch_standardized_filings,
)
from app.institutional_tools import (
    analyze_block_trade_volume,
    fetch_insider_transactions,
    fetch_institutional_holdings,
)
from app.macro_tools import (
    analyze_sector_momentum,
    fetch_fred_macro_data,
    fetch_geopolitical_risk_signals,
    fetch_india_macro_data,
)
from app.quant_tools import (
    calculate_factor_information_coefficient,
    fetch_historical_prices,
    test_pairs_cointegration,
    validate_signal_statistics,
)
from app.risk_tools import (
    analyze_debt_maturity_wall,
    calculate_tail_risk_metrics,
    extract_concentration_risks,
    run_monte_carlo_stress_test,
)
from app.screener_tools import screen_nse_stocks
from app.sentiment_tools import (
    analyze_management_tone,
    calculate_finbert_sentiment,
    extract_catalyst_events,
    fetch_text_corpus,
)
from app.valuation_tools import (
    calculate_absolute_dcf,
    calculate_graham_intrinsic_value,
    calculate_margin_of_safety,
    calculate_relative_multiples,
)

# ---------------------------------------------------------------------
# 1. Pydantic Schemas for Multi-Agent Communication & State
# ---------------------------------------------------------------------


class SectorCheck(BaseModel):
    is_specified: bool = Field(
        description="True if a specific sector, industry, or company/ticker is mentioned in the query."
    )
    sector_or_company: str | None = Field(
        None, description="The identified sector or company if specified."
    )
    query_type: Literal["single_asset", "screening"] = Field(
        default="single_asset",
        description=(
            "'screening' when the user wants recommendations or asks which companies to invest in. "
            "'single_asset' when the user names a specific ticker, company, or sector to analyze."
        ),
    )
    market: Literal["IN", "US", "GLOBAL"] = Field(
        default="GLOBAL",
        description="Detected target market: 'IN' for India (rupees/NSE/BSE/Indian companies mentioned), 'US' for US market, 'GLOBAL' otherwise.",
    )
    capital_inr: float | None = Field(
        default=None,
        description="Investment capital extracted from the query in INR. E.g. '5000 rupees' → 5000.0.",
    )
    risk_profile: str | None = Field(
        default=None,
        description="Risk appetite extracted from query: e.g. 'high', 'moderate', 'conservative', 'decent'.",
    )


class InvestmentProposal(BaseModel):
    ticker: str = Field(description="Stock ticker symbol (e.g. AAPL, GOOG).")
    sector: str = Field(description="The sector of the company.")
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="Proposed investment action."
    )
    price_target: float = Field(description="Target stock price.")
    rationale: str = Field(description="Brief logic for the proposal.")


class StatisticalArbitrage(BaseModel):
    tested_pair: str = Field(description="Asset vs Sector ETF")
    is_cointegrated: bool = Field(description="Is cointegrated")
    spread_z_score: float = Field(description="Spread Z-score")
    adf_p_value: float = Field(description="ADF p-value")


class FactorAnalysis(BaseModel):
    primary_factor_ic: float = Field(description="Primary factor IC")
    factor_predictive_power: Literal["STRONG", "WEAK", "NOISE"] = Field(
        description="Predictive power"
    )


class SignalValidation(BaseModel):
    t_test_p_value: float = Field(description="T-test p-value")
    statistical_significance: Literal["SIGNIFICANT", "INSIGNIFICANT"] = Field(
        description="Statistical significance"
    )
    noise_warning: str = Field(
        description="String explaining if the recent price move is just variance."
    )


class QuantAnalysis(BaseModel):
    statistical_arbitrage: StatisticalArbitrage = Field(
        description="Statistical arbitrage spread analysis."
    )
    factor_analysis: FactorAnalysis = Field(
        description="Factor analysis information coefficient details."
    )
    signal_validation: SignalValidation = Field(
        description="Welch's t-test signal validation."
    )
    quantitative_confidence_score: int = Field(
        description="Quantitative confidence score (1-100)."
    )


class ForensicAnalysis(BaseModel):
    altman_z_score: float | None = Field(
        default=None, description="The Altman Z-Score calculated for the asset."
    )
    bankruptcy_risk: Literal["LOW", "GREY", "HIGH", "UNKNOWN"] = Field(
        default="UNKNOWN", description="Bankruptcy risk assessment based on Altman Z-Score."
    )
    piotroski_f_score: int | None = Field(
        default=None, description="Piotroski F-Score (range 0-9) evaluating operational health."
    )
    beneish_m_score: float | None = Field(
        default=None, description="Beneish M-Score evaluating earnings manipulation probability."
    )
    earnings_manipulation_flag: bool = Field(
        default=False, description="True if Beneish M-Score is greater than -1.78, indicating high probability of earnings manipulation."
    )


class EarningsQuality(BaseModel):
    sloan_accrual_ratio: float | None = Field(
        default=None, description="Sloan Accrual Ratio. Positive values mean net income is higher than cash flow."
    )
    cash_flow_alignment: Literal["STRONG", "WEAK"] = Field(
        default="WEAK", description="Cash flow alignment level based on the Sloan Accrual Ratio."
    )


class FinancialAnalysis(BaseModel):
    fundamental_health_score: int = Field(
        description="A fundamental health score from 1 (poor) to 10 (excellent)."
    )
    forensic_analysis: ForensicAnalysis = Field(
        description="Detailed forensic accounting metrics."
    )
    earnings_quality: EarningsQuality = Field(
        description="Analysis of earnings quality and accruals."
    )
    graham_defensive_pass: bool = Field(
        description="True if the asset passes Benjamin Graham's strict defensive investor criteria."
    )
    critical_accounting_red_flags: list[str] = Field(
        description="Flagged accounting abnormalities or risks."
    )
    fundamental_strengths: list[str] = Field(
        description="Key strengths and positive forensic signals."
    )


class InterestRateRegime(BaseModel):
    current_10yr_yield: float = Field(description="The current 10-year Treasury Yield.")
    trend: Literal["RISING", "FALLING", "FLAT", "NEUTRAL"] = Field(
        description="Trend of the 10yr yield (RISING, FALLING, FLAT, NEUTRAL)."
    )
    impact_on_sector: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(
        description="Impact of the interest rate regime on the sector."
    )


class IndustryMomentum(BaseModel):
    sector_etf_performance_6m: float = Field(
        description="6-month performance of the sector ETF in percentage."
    )
    relative_to_sp500: Literal["OUTPERFORMING", "UNDERPERFORMING", "MATCHING"] = Field(
        description="Performance relative to SPY."
    )
    lifecycle_stage: Literal["GROWTH", "MATURE", "DECLINE", "ACCELERATING"] = Field(
        description="Sector lifecycle stage."
    )


class MacroAnalysis(BaseModel):
    macro_environment_status: Literal["TAILWIND", "HEADWIND", "NEUTRAL"] = Field(
        description="Overall macroeconomic status."
    )
    interest_rate_regime: InterestRateRegime = Field(
        description="Current interest rate environment detail."
    )
    industry_momentum: IndustryMomentum = Field(
        description="Industry sector momentum details."
    )
    geopolitical_and_regulatory_risks: list[str] = Field(
        description="List of external geopolitical trade or regulatory risks."
    )
    macro_confidence_score: int = Field(description="Overall confidence score (1-100).")


class NewsSentimentDetails(BaseModel):
    finbert_polarity_score: float = Field(
        description="Sentiment polarity score from FinBERT analysis."
    )
    retail_hype_index: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Retail hype index level."
    )
    institutional_analyst_consensus: Literal["BUY", "SELL", "HOLD"] = Field(
        description="Institutional analyst consensus."
    )


class ManagementToneAnalysis(BaseModel):
    uncertainty_word_count_change: str = Field(
        description="Percentage change in uncertainty words (e.g. +15% or -5%)."
    )
    litigation_risk_mentions: int = Field(
        description="Number of litigation risk mentions found in the text."
    )
    overall_confidence_trend: Literal["IMPROVING", "DEGRADING", "STABLE"] = Field(
        description="Trend of management confidence."
    )
    key_quote_flagged: str = Field(
        description="Quote from the transcript showing hedging or forward guidance."
    )


class CatalystEvent(BaseModel):
    event_description: str = Field(description="Description of the catalyst event.")
    expected_timeframe: str = Field(description="Expected timeframe for the event.")
    impact_probability: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        description="Probability of impact."
    )


class SentimentAnalysis(BaseModel):
    overall_sentiment_status: Literal["BULLISH", "BEARISH", "MIXED"] = Field(
        description="Overall sentiment status."
    )
    news_sentiment: NewsSentimentDetails = Field(
        description="Detailed news media sentiment metrics."
    )
    management_tone_analysis: ManagementToneAnalysis = Field(
        description="Earnings transcript management tone metrics."
    )
    identified_catalysts: list[CatalystEvent] = Field(
        description="Identified future catalyst events."
    )
    sentiment_confidence_score: int = Field(
        description="Sentiment confidence score (1-100)."
    )


class IntrinsicValue(BaseModel):
    dcf_implied_value: float = Field(
        description="Discounted Cash Flow implied intrinsic value."
    )
    graham_formula_value: float = Field(
        description="Benjamin Graham's revised formula intrinsic value."
    )
    blended_fair_value: float = Field(
        description="Blended fair value estimate (e.g. weighted average of DCF and Graham)."
    )


class RelativeMultiples(BaseModel):
    pe_ratio: float | None = Field(None, description="Price-to-Earnings Ratio.")
    peg_ratio: float | None = Field(None, description="PEG Ratio.")
    ev_to_ebitda: float | None = Field(None, description="Enterprise Value to EBITDA.")
    sector_relative_valuation: str = Field(
        description="Relative valuation ranking against industry peers (e.g. CHEAPER_THAN_80_PERCENT_OF_PEERS)."
    )


class ValuationAnalysis(BaseModel):
    valuation_status: Literal["UNDERVALUED", "FAIRLY_VALUED", "OVERVALUED"] = Field(
        description="The overall valuation status."
    )
    current_price: float = Field(description="The current market price of the asset.")
    intrinsic_value: IntrinsicValue = Field(
        description="Absolute intrinsic value breakdown."
    )
    margin_of_safety_percentage: float = Field(
        description="Percentage margin of safety."
    )
    relative_multiples: RelativeMultiples = Field(
        description="Relative valuation multiples and ranking."
    )
    valuation_risks: list[str] = Field(
        description="Key valuation sensitivities and risks."
    )


class QuantitativeRisk(BaseModel):
    value_at_risk_95: float = Field(
        description="Historical daily Value at Risk at 95% confidence level in percentage (e.g. -4.20)."
    )
    conditional_var_95: float = Field(
        description="Historical daily Conditional Value at Risk at 95% confidence level in percentage (e.g. -6.80)."
    )
    max_historical_drawdown_percentage: float = Field(
        description="Maximum historical drawdown in percentage."
    )
    monte_carlo_probability_of_20_percent_loss: float = Field(
        description="Probability of 20%+ loss over the next year from Monte Carlo simulation (0.0 to 1.0)."
    )


class RiskAnalysis(BaseModel):
    overall_risk_rating: Literal["EXTREME", "HIGH", "MODERATE", "LOW"] = Field(
        description="Overall risk rating."
    )
    quantitative_risk: QuantitativeRisk = Field(
        description="Quantitative tail risk metrics."
    )
    structural_vulnerabilities: list[str] = Field(
        description="Flagged structural vulnerabilities (e.g., customer concentration, debt maturity walls)."
    )
    thesis_invalidation_catalysts: list[str] = Field(
        description="Catalyst events that would invalidate the investment thesis."
    )
    risk_confidence_score: int = Field(
        description="Risk analysis confidence score (1-100)."
    )


class InsiderActivity(BaseModel):
    net_buying_selling_ratio_90d: float | None = Field(
        None, description="Net buying to selling ratio over 90 days."
    )
    key_insider_buys: list[str] = Field(
        description="Key open market purchases flagged."
    )
    verdict: Literal["BULLISH", "BEARISH", "NEUTRAL", "DATA_UNAVAILABLE"] = Field(
        description="Insider transaction sentiment verdict."
    )


class HedgeFundActivity(BaseModel):
    institutional_ownership_percentage: float | None = Field(
        None, description="Hedge fund/institutional ownership percentage."
    )
    quarter_over_quarter_change: str | None = Field(
        None, description="QoQ change in institutional holdings (e.g. +4.2%)."
    )
    top_buyers: list[str] = Field(description="Top institutional buyers.")


class InstitutionalAnalysis(BaseModel):
    smart_money_sentiment: Literal["ACCUMULATION", "DISTRIBUTION", "NEUTRAL", "DATA_UNAVAILABLE"] = Field(
        description="Overall smart money status."
    )
    insider_activity: InsiderActivity = Field(
        description="Detailed insider transaction analytics."
    )
    hedge_fund_activity: HedgeFundActivity = Field(
        description="Hedge fund 13F ownership details."
    )
    block_trade_anomalies: list[str] = Field(
        description="Flagged dark pool block trade anomalies."
    )
    flow_confidence_score: int = Field(
        description="Order flow confidence score (1-100)."
    )


class ConsensusCheck(BaseModel):
    consensus_reached: bool = Field(
        description="True if the committee agrees on action and target."
    )
    final_decision: Literal["BUY", "SELL", "HOLD"] = Field(
        description="The finalized action."
    )
    rationale: str = Field(
        description="The detailed rationale from consensus evaluation."
    )
    revised_proposal: InvestmentProposal | None = Field(
        None, description="A revised proposal if consensus is not reached."
    )


class WorkflowState(BaseModel):
    query: str | None = None
    sector: str | None = None
    proposal: dict | None = None
    quant_analysis: dict | None = None
    financial_analysis: dict | None = None
    macro_analysis: dict | None = None
    sentiment_analysis: dict | None = None
    institutional_analysis: dict | None = None
    valuation_analysis: dict | None = None
    risk_analysis: dict | None = None
    consensus: dict | None = None
    debate_turn_count: int = 0
    security_event: bool = False
    security_reason: str | None = None
    revised_valuation: dict | None = None
    debate_triggered: bool | None = None
    # Indian market / screening context
    query_type: str | None = None        # "single_asset" | "screening"
    market: str | None = None            # "IN" | "US" | "GLOBAL"
    capital_inr: float | None = None     # investment amount in INR
    risk_profile: str | None = None      # "conservative" | "moderate" | "high" etc.
    screened_ticker: str | None = None   # ticker auto-picked by stock_screener



# ---------------------------------------------------------------------
# 2. LLM Configuration
# ---------------------------------------------------------------------

project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "learning-agents-for-finance")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true"

if use_vertex:
    # Vertex AI Premium — full resource path; model: gemini-2.5-flash
    model_path = f"projects/{project_id}/locations/{location}/publishers/google/models/gemini-2.5-flash"
else:
    # AI Studio fallback (requires GEMINI_API_KEY or GOOGLE_API_KEY)
    model_path = "gemini-2.5-flash"

llm = Gemini(
    model=model_path,
    retry_options=types.HttpRetryOptions(attempts=3),
)

# ---------------------------------------------------------------------
# 3. LLM Agents Definitions
# ---------------------------------------------------------------------

sector_detector = LlmAgent(
    name="sector_detector",
    model=llm,
    instruction=(
        "Analyze the user query and extract the following fields:\n"
        "1. is_specified: True ONLY if the user names a specific ticker (e.g. INFY, MSFT), company name, or sector. False if it's an open question.\n"
        "2. sector_or_company: The named entity if is_specified is True, else null.\n"
        "3. query_type: Set to 'screening' if the user asks which/best/recommend/where to invest. Set to 'single_asset' if they name a specific asset.\n"
        "4. market: 'IN' if the query mentions rupees (₹/INR/Rs), NSE, BSE, Indian companies, or India. 'US' if USD/S&P/NYSE mentioned. Otherwise 'GLOBAL'.\n"
        "5. capital_inr: Extract numeric investment amount in INR if mentioned (e.g. '5000 rupees' → 5000.0, '10k' → 10000.0). Null if not mentioned.\n"
        "6. risk_profile: Extract risk appetite if mentioned (e.g. 'decent', 'high', 'aggressive', 'conservative', 'moderate'). Null if not mentioned.\n"
        "Return the structured classification."
    ),
    output_schema=SectorCheck,
)

proposal_generator = LlmAgent(
    name="proposal_generator",
    model=llm,
    instruction=(
        "Generate an initial investment proposal based on the query and identified sector/company/screened ticker. "
        "If market context is Indian (ticker ends in .NS or .BO), use INR for price targets, reference NSE/BSE norms. "
        "Outline ticker (use .NS suffix for Indian stocks), sector, action (BUY/SELL/HOLD), price_target (in the local currency), and rationale."
    ),
    output_schema=InvestmentProposal,
)

stock_screener = LlmAgent(
    name="stock_screener",
    model=llm,
    instruction=(
        "You are a quantitative stock screening expert for Indian equity markets. "
        "Use the screen_nse_stocks tool to fetch a live shortlist of the best NSE-listed companies "
        "matching the user's criteria (sector, risk profile, index universe). "
        "The tool fetches real-time data from NSE India's public index files and scores each company "
        "using fundamental ratios (P/E, P/B, ROE, debt/equity, 52-week return). "
        "After receiving the tool output, select the SINGLE BEST candidate from the returned list "
        "based on the composite_score and the user's stated risk profile. "
        "If risk_profile is 'high' or 'aggressive', prefer higher ROE and 52w momentum. "
        "If 'conservative', prefer low P/E and low debt. "
        "Output ONLY the ticker symbol of your chosen stock (e.g. 'INFY.NS') — "
        "no JSON, no explanation, no markdown."
    ),
    tools=[screen_nse_stocks],
)

quant_agent = LlmAgent(
    name="quant_agent",
    model=llm,
    instruction=(
        "You are the Lead Quantitative Analyst on an institutional investment committee. "
        "Your mandate is to evaluate statistical anomalies, factor exposures, and pair-trading cointegration. "
        "You do not look at standard charts. You must use your provided Python tools to run Augmented Dickey-Fuller tests for mean reversion, "
        "evaluate Alpha factors using the Information Coefficient (IC), and validate all trading signals strictly using p-values and t-tests. "
        "If a signal does not possess a p-value < 0.05, you must flag it as statistically insignificant noise. "
        "Output your findings strictly in the required JSON schema."
    ),
    tools=[
        test_pairs_cointegration,
        calculate_factor_information_coefficient,
        validate_signal_statistics,
    ],
    output_schema=QuantAnalysis,
    output_key="quant_analysis",
)

financial_agent = LlmAgent(
    name="financial_agent",
    model=llm,
    instruction=(
        "You are the Lead Fundamental Analyst on an institutional investment committee. "
        "Your role is forensic accounting and fundamental health screening. Do not trust management's adjusted EBITDA or narrative. "
        "You will use your provided Python tools to pull standardized SEC filings and run them through rigorous historical models: "
        "the Piotroski F-Score for operational efficiency, the Altman Z-Score for bankruptcy risk, and the Beneish M-Score for earnings manipulation. "
        "Furthermore, evaluate the asset against Benjamin Graham's strict defensive investor criteria. "
        "Synthesize these deterministic outputs into a final fundamental health score and flag any accounting anomalies.\n\n"
        "CRITICAL SCHEMA COMPLIANCE RULE: "
        "If any tool returns {'status': 'error', ...} or returns incomplete data (e.g. a micro-cap with no SEC filings, "
        "an Indian stock with limited yfinance history), you MUST still populate EVERY field in the output schema. "
        "Use the following fallback values for missing data:\n"
        "  - Numeric fields (altman_z_score, piotroski_f_score, beneish_m_score, sloan_accrual_ratio, fundamental_health_score): null\n"
        "  - Categorical fields (bankruptcy_risk, cash_flow_alignment): 'UNKNOWN'\n"
        "  - Boolean flags (earnings_manipulation_flag, graham_defensive_pass): false\n"
        "  - String fields (anomaly_flags): 'Data unavailable — tool returned error'\n"
        "NEVER omit a field. A null value is valid JSON. A missing key will crash the pipeline."
    ),
    tools=[
        fetch_standardized_filings,
        compute_forensic_scores,
        analyze_earnings_quality,
        evaluate_graham_criteria,
    ],
    output_schema=FinancialAnalysis,
    output_key="financial_analysis",
)

macro_agent = LlmAgent(
    name="macro_agent",
    model=llm,
    instruction=(
        "You are the Lead Macroeconomist and Industry Strategist on an institutional investment committee. "
        "Your mandate is to assess the external environment surrounding a specific asset. "
        "You are strictly forbidden from guessing inflation rates, interest rates, or geopolitical events. "
        "You must use your provided Python tools to fetch hard data from central bank databases and sector performance trackers. "
        "For Indian stocks (.NS/.BO tickers): use fetch_india_macro_data for RBI monetary context, INR/USD trend, "
        "and Nifty 50 market regime; use analyze_sector_momentum with the relevant NSE sector index (^CNXIT for IT, "
        "^NSEBANK for Banking, ^CNXPHARMA for Pharma, ^CNXAUTO for Auto, etc.) and benchmark='^NSEI'. "
        "For US stocks: use fetch_fred_macro_data for Fed rates and CPI; use analyze_sector_momentum with US sector ETFs (XLK, XLE, etc.). "
        "Evaluate the current interest rate regime, sector momentum, and geopolitical trade risks. "
        "Synthesize this data to determine if the macroeconomic environment provides a Tailwind, Headwind, or Neutral effect on the asset. "
        "Output your findings strictly in the required JSON schema."
    ),
    tools=[
        fetch_fred_macro_data,
        fetch_india_macro_data,
        analyze_sector_momentum,
        fetch_geopolitical_risk_signals,
    ],
    output_schema=MacroAnalysis,
    output_key="macro_analysis",
)

sentiment_agent = LlmAgent(
    name="sentiment_agent",
    model=llm,
    instruction=(
        "You are the Lead Behavioral Analyst and NLP Specialist on an institutional investment committee. "
        "Your mandate is to quantify the market narrative and management tone surrounding an asset. "
        "You are strictly forbidden from subjectively interpreting news headlines based on your own bias. "
        "You must use your provided Python NLP tools to fetch earnings transcripts, press releases, and financial news, and run them through FinBERT and financial-specific dictionaries to score sentiment polarity. "
        "Identify any critical catalysts and flag 'hedging' language used by management. "
        "Synthesize these quantitative text metrics to determine if the prevailing sentiment is Bullish, Bearish, or Divergent. "
        "Output your findings strictly in the required JSON schema."
    ),
    tools=[
        fetch_text_corpus,
        calculate_finbert_sentiment,
        analyze_management_tone,
        extract_catalyst_events,
    ],
    output_schema=SentimentAnalysis,
    output_key="sentiment_analysis",
)

institutional_agent = LlmAgent(
    name="institutional_agent",
    model=llm,
    instruction=(
        "You are the Institutional Order Flow Analyst. Your mandate is to track 'Smart Money'. "
        "You are strictly forbidden from analyzing social media or unverified rumors. "
        "You must use your provided Python tools to query SEC EDGAR databases for Form 13F (Hedge Fund Holdings) "
        "and Form 4 (Insider Transactions), as well as track large dark-pool block trades. "
        "Synthesize this regulatory data to determine if institutions and insiders are in an Accumulation or Distribution phase.\n\n"
        "CRITICAL REGULATORY RULE FOR NON-US MARKETS: "
        "SEC Form 13F and Form 4 filings are US-only regulatory disclosures. For Indian stocks (.NS/.BO suffixes) "
        "or other non-US stocks where the tools return status success but indicate disclosures are unavailable "
        "(or return DATA_UNAVAILABLE / None), you MUST honestly report this in the output schema:\n"
        "  - Set smart_money_sentiment to 'DATA_UNAVAILABLE'\n"
        "  - Set insider_activity.net_buying_selling_ratio_90d to null/None\n"
        "  - Set insider_activity.verdict to 'DATA_UNAVAILABLE'\n"
        "  - Set hedge_fund_activity.institutional_ownership_percentage to null/None\n"
        "  - Set hedge_fund_activity.quarter_over_quarter_change to null/None\n"
        "Do NOT invent placeholder values or default to 'NEUTRAL' if the data is genuinely unavailable."
    ),
    tools=[
        fetch_insider_transactions,
        fetch_institutional_holdings,
        analyze_block_trade_volume,
    ],
    output_schema=InstitutionalAnalysis,
    output_key="institutional_analysis",
)

valuation_agent = LlmAgent(
    name="valuation_agent",
    model=llm,
    instruction="""You are the Valuation Expert on an institutional investment committee. Your sole mandate is to determine the fair intrinsic value of an asset and assess whether the current market price offers a Margin of Safety. You are strictly forbidden from guessing valuations or calculating math manually. You must use your provided Python tools to run Discounted Cash Flow (DCF) models, calculate Benjamin Graham's intrinsic value formula, and rank the asset's multiples against its industry peers. Combine these absolute and relative valuation models to determine if the asset is Undervalued, Fairly Valued, or Overvalued. Output your final appraisal strictly in the required JSON schema.""",
    output_schema=ValuationAnalysis,
    output_key="valuation_analysis",
    tools=[
        fetch_standardized_filings,
        calculate_absolute_dcf,
        calculate_graham_intrinsic_value,
        calculate_relative_multiples,
        calculate_margin_of_safety,
    ],
)

risk_agent = LlmAgent(
    name="risk_agent",
    model=llm,
    instruction=(
        "You are the Chief Risk Officer on an institutional investment committee. "
        "Your exclusive mandate is downside protection. You must assume every investment thesis is overly optimistic and actively search for ways this asset could cause catastrophic portfolio damage. "
        "You are forbidden from guessing risk metrics. "
        "You will use your provided Python tools to calculate Value at Risk (VaR), Conditional Value at Risk (CVaR), run Monte Carlo stress tests, and parse SEC filings for concentration risks (e.g., reliance on a single customer).\n\n"
        "To perform these calculations, call calculate_tail_risk_metrics and analyze_debt_maturity_wall directly for the requested ticker.\n"
        "Synthesize these vulnerabilities to determine the asset's structural fragility. "
        "Output your findings strictly in the required JSON schema."
    ),
    tools=[
        calculate_tail_risk_metrics,
        run_monte_carlo_stress_test,
        analyze_debt_maturity_wall,
        extract_concentration_risks,
    ],
    output_schema=RiskAnalysis,
    output_key="risk_analysis",
)

consensus_evaluator = LlmAgent(
    name="consensus_evaluator",
    model=llm,
    instruction="Evaluate the consensus between the initial proposal, valuation, and risk analysis. If consensus is reached, set consensus_reached=True. If not, set consensus_reached=False and provide a revised proposal addressing the concerns.",
    output_schema=ConsensusCheck,
)

# ---------------------------------------------------------------------
# 4. Security Guardrail Functions
# ---------------------------------------------------------------------


def is_security_violation(text: str) -> tuple[bool, str]:
    """Inspects query text for PII leaks or rule override prompt injections."""
    text_lower = text.lower()

    # 1. Prompt Injection & Override Detection
    override_keywords = [
        "ignore all previous",
        "ignore previous instructions",
        "system prompt",
        "bypass rule",
        "auto-approve",
        "auto approve",
        "force approval",
        "bypass the rules",
        "override instructions",
        "you must approve",
        "developer mode",
    ]
    for kw in override_keywords:
        if kw in text_lower:
            return (
                True,
                f"Prompt injection/override instruction detected (keyword: '{kw}').",
            )

    # 2. PII Detection: Email
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    if re.search(email_pattern, text):
        return True, "PII leak detected: Email address."

    # 3. PII Detection: Phone Number
    phone_pattern = r"(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}"
    if re.search(phone_pattern, text):
        return True, "PII leak detected: Phone number."

    # 4. PII Detection: SSN
    ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
    if re.search(ssn_pattern, text):
        return True, "PII leak detected: Social Security Number."

    return False, ""


def _to_dict(obj: Any) -> dict:
    """Safely converts Pydantic models, objects, or JSON strings into dictionaries."""
    if not obj:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, str):
        try:
            import json
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _normalize_proposal_ticker(proposal: Any, market: str | None) -> Any:
    """Helper to ensure Indian stock tickers have the correct .NS suffix for yfinance."""
    if not proposal:
        return proposal
    if isinstance(proposal, dict):
        ticker = proposal.get("ticker", "")
        if market == "IN" and ticker and not ticker.endswith((".NS", ".BO")):
            proposal["ticker"] = f"{ticker.upper()}.NS"
    else:
        ticker = getattr(proposal, "ticker", "")
        if market == "IN" and ticker and not ticker.endswith((".NS", ".BO")):
            proposal.ticker = f"{ticker.upper()}.NS"
    return proposal


# ---------------------------------------------------------------------
# 5. Function Nodes (Coordination, Input Resumption, Security, Formatting)
# ---------------------------------------------------------------------


@node(rerun_on_resume=True)
async def portfolio_manager(ctx: Context, node_input: Any):
    # Determine the query
    if hasattr(node_input, "parts") and node_input.parts:
        ctx.state["query"] = node_input.parts[0].text
    elif isinstance(node_input, str):
        ctx.state["query"] = node_input
    elif isinstance(node_input, dict) and "query" in node_input:
        ctx.state["query"] = node_input["query"]

    query = ctx.state.get("query", "")

    # Run non-LLM security check first on the raw query
    violation, reason = is_security_violation(query)
    if violation:
        ctx.state["security_event"] = True
        ctx.state["security_reason"] = reason
        yield Event(output=query, route="security_flagged")
        return

    # Handle sector checks & interactive prompt
    sector = ctx.state.get("sector")
    if not sector:
        if ctx.resume_inputs and "get_sector" in ctx.resume_inputs:
            raw_val = ctx.resume_inputs["get_sector"]
            if isinstance(raw_val, dict):
                sector = raw_val.get("get_sector") or raw_val.get("output") or ""
            else:
                sector = str(raw_val)
            ctx.state["sector"] = sector
        else:
            detection: dict = await ctx.run_node(sector_detector, node_input=query)
            query_type = detection.get("query_type", "single_asset")
            market = detection.get("market", "GLOBAL")
            capital = detection.get("capital_inr")
            risk_profile = detection.get("risk_profile")

            # Persist context for downstream nodes
            ctx.state["query_type"] = query_type
            ctx.state["market"] = market
            if capital is not None:
                ctx.state["capital_inr"] = capital
            if risk_profile:
                ctx.state["risk_profile"] = risk_profile

            if detection.get("is_specified") and detection.get("sector_or_company"):
                # Named asset — proceed directly
                sector = detection.get("sector_or_company")
                ctx.state["sector"] = sector
            elif query_type == "screening":
                # Discovery query: auto-screen instead of asking the user
                screener_prompt = (
                    f"Market: {market}\n"
                    f"Sector preference: {detection.get('sector_or_company') or 'open-ended'}\n"
                    f"Capital: {f'₹{capital:,.0f}' if capital else 'unspecified'}\n"
                    f"Risk profile: {risk_profile or 'moderate'}\n"
                    f"User query: {query}"
                )
                screened_ticker: str = await ctx.run_node(
                    stock_screener, node_input=screener_prompt
                )
                # Normalise: strip whitespace/quotes the LLM might add
                screened_ticker = str(screened_ticker).strip().strip('"').strip("'")
                sector = screened_ticker
                ctx.state["sector"] = sector
                ctx.state["screened_ticker"] = screened_ticker
            else:
                # Single-asset but not named — ask the user
                yield RequestInput(
                    interrupt_id="get_sector",
                    message="Please specify the company, ticker, or sector you want to analyze (e.g. 'INFY.NS', 'Banking sector', 'Technology').",
                )
                return

    # Determine if we are looping back after parallel execution
    is_loop = all(
        ctx.state.get(k) is not None
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

    if is_loop:
        # Retrieve all accumulated data safely coerced to dicts
        proposal_dict = _to_dict(ctx.state.get("proposal"))
        quant_dict = _to_dict(ctx.state.get("quant_analysis"))
        financial_dict = _to_dict(ctx.state.get("financial_analysis"))
        macro_dict = _to_dict(ctx.state.get("macro_analysis"))
        sentiment_dict = _to_dict(ctx.state.get("sentiment_analysis"))
        institutional_dict = _to_dict(ctx.state.get("institutional_analysis"))
        valuation_dict = _to_dict(ctx.state.get("valuation_analysis"))
        risk_dict = _to_dict(ctx.state.get("risk_analysis"))

        # Synthesis Check Logic Gate: Detect major contradictions
        valuation_status = (valuation_dict or {}).get("valuation_status", "")
        risk_rating = (risk_dict or {}).get("overall_risk_rating", "")
        proposal_action = (proposal_dict or {}).get("action", "")

        is_contradiction = False
        if valuation_status == "UNDERVALUED" and risk_rating in ("HIGH", "EXTREME"):
            is_contradiction = True
        elif proposal_action == "BUY" and (
            risk_rating in ("HIGH", "EXTREME") or valuation_status == "OVERVALUED"
        ):
            is_contradiction = True

        debate_turn = ctx.state.get("debate_turn_count", 0)

        if is_contradiction and debate_turn == 0:
            ctx.state["debate_turn_count"] = debate_turn + 1
            yield Event(output=risk_dict, route="trigger_debate")
            return
        else:
            active_val = ctx.state.get("revised_valuation") or valuation_dict
            consensus_context = f"""
Current Proposal: {proposal_dict}
Quant Analysis: {quant_dict}
Financial Analysis: {financial_dict}
Macro Analysis: {macro_dict}
Sentiment Analysis: {sentiment_dict}
Institutional Flow Analysis: {institutional_dict}
Valuation Analysis: {active_val}
Risk Analysis: {risk_dict}
"""
            consensus_check = await ctx.run_node(
                consensus_evaluator, node_input=consensus_context
            )
            if consensus_check and getattr(consensus_check, "revised_proposal", None):
                consensus_check.revised_proposal = _normalize_proposal_ticker(
                    consensus_check.revised_proposal, ctx.state.get("market")
                )
            ctx.state["consensus"] = consensus_check
            yield Event(output=consensus_check, route="consensus_reached")
            return
    else:
        # First execution run: generate the initial proposal
        market = ctx.state.get("market", "GLOBAL")
        capital = ctx.state.get("capital_inr")
        risk_profile = ctx.state.get("risk_profile")
        screened = ctx.state.get("screened_ticker")

        proposal_prompt = (
            f"User Query: {query}\n"
            f"Sector/Company/Ticker: {sector}\n"
            f"Market: {market}\n"
            + (f"Investment Capital: ₹{capital:,.0f} INR\n" if capital else "")
            + (f"Risk Profile: {risk_profile}\n" if risk_profile else "")
            + ("Note: This ticker was auto-screened based on user's open-ended request.\n" if screened else "")
        )
        proposal = await ctx.run_node(
            proposal_generator, node_input=proposal_prompt
        )
        proposal = _normalize_proposal_ticker(proposal, market)
        ctx.state["proposal"] = proposal
        yield Event(output=proposal, route="continue")
        return


@node(rerun_on_resume=True)
async def human_review(ctx: Context, node_input: Any):
    reason = ctx.state.get("security_reason", "Unknown violation")
    query = ctx.state.get("query", "")

    # Check if the administrator responded to override or reject
    if ctx.resume_inputs and "human_override" in ctx.resume_inputs:
        raw_val = ctx.resume_inputs["human_override"]
        if isinstance(raw_val, dict):
            decision = (
                (raw_val.get("human_override") or raw_val.get("output") or "")
                .strip()
                .lower()
            )
        else:
            decision = str(raw_val).strip().lower()
        if decision in ("approve", "allow"):
            # Bypass warning: continue to generating initial proposal
            ctx.state["sector"] = "open-ended"
            proposal_prompt = f"User Query: {query}\nSector/Company Context: open-ended"
            proposal = await ctx.run_node(
                proposal_generator, node_input=proposal_prompt
            )
            proposal = _normalize_proposal_ticker(proposal, ctx.state.get("market"))
            ctx.state["proposal"] = proposal
            ctx.state["iteration"] = 1
            yield Event(output=proposal, route="continue")
            return
        else:
            # Manual block
            block_msg = f"### ❌ SECURITY BLOCKED\n\nThis query was blocked by administrator review.\nReason: {reason}"
            yield Event(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text=block_msg)]
                )
            )
            blocked_consensus = {
                "consensus_reached": True,
                "final_decision": "UNDER_REVIEW",
                "rationale": f"Blocked by administrator review: {reason}",
                "revised_proposal": None,
            }
            yield Event(output=blocked_consensus, route="exit")
            return

    yield RequestInput(
        interrupt_id="human_override",
        message=f"⚠️ [SECURITY ALERT] Potential PII leak or prompt injection attempt detected!\n"
        f'Query: "{query}"\n'
        f"Reason: {reason}\n\n"
        f"Please review. Type 'approve' to bypass security and continue, or 'reject' to block.",
    )


@node(rerun_on_resume=True)
async def debate_node(ctx: Context, node_input: Any):
    valuation_dict = _to_dict(ctx.state.get("valuation_analysis"))
    risk_dict = _to_dict(ctx.state.get("risk_analysis"))

    # Format CRO concern and details
    vulnerabilities_list = "\n".join(
        [f"- {v}" for v in (risk_dict or {}).get("structural_vulnerabilities", [])]
    )

    debate_trigger_msg = (
        "### ⚖️ DEBATE MODE ACTIVATED (Strict 1-Turn Limit)\n\n"
        "**Portfolio Manager (CIO):** A discrepancy has been detected. Valuation is bullish/undervalued, "
        f"but Risk is rated **{(risk_dict or {}).get('overall_risk_rating')}**. Valuation Expert, please review "
        "the Risk Analyst's structural objections and adjust/defend your position.\n\n"
        f"**CRO (Risk Analyst) structural vulnerabilities:**\n"
        f"{vulnerabilities_list}\n\n"
        f"**CRO Tail Risk Metrics:**\n"
        f"- Historical VaR 95%: **{(risk_dict or {}).get('quantitative_risk', {}).get('value_at_risk_95')}%**\n"
        f"- Historical CVaR 95%: **{(risk_dict or {}).get('quantitative_risk', {}).get('conditional_var_95')}%**\n"
        f"- Max Historical Drawdown: **{(risk_dict or {}).get('quantitative_risk', {}).get('max_historical_drawdown_percentage')}%**\n\n"
        "**Challenging the Valuation Expert...**"
    )
    yield Event(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=debate_trigger_msg)]
        )
    )

    # Pass the challenge to the valuation_agent
    debate_prompt = (
        f"You are the Valuation Expert. The Chief Risk Officer has raised significant objections. "
        f"Here is the Risk Analyst's report: {risk_dict}. "
        f"Your current valuation analysis was: {valuation_dict}. "
        f"Please adjust your valuation, or defend your original score. Return your final appraisal."
    )

    revised_valuation = await ctx.run_node(valuation_agent, node_input=debate_prompt)
    revised_valuation = _to_dict(revised_valuation)
    ctx.state["revised_valuation"] = revised_valuation
    ctx.state["debate_triggered"] = True

    curr = "₹" if ctx.state.get("market") == "IN" else "$"

    debate_outcome_msg = (
        "### 💬 VALUATION DEBATE RESPONSE\n\n"
        f"**Valuation Expert Response:**\n"
        f"- Revised Valuation Status: **{revised_valuation.get('valuation_status')}**\n"
        f"- Revised Blended Fair Value: **{curr}{revised_valuation.get('intrinsic_value', {}).get('blended_fair_value')}**\n"
        f"- Margin of Safety: **{revised_valuation.get('margin_of_safety_percentage')}%**\n"
        f"- Sensitivities and Risks Addressed: {revised_valuation.get('valuation_risks')}"
    )
    yield Event(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=debate_outcome_msg)]
        )
    )

    # Prepare final consensus evaluation
    proposal_dict = _to_dict(ctx.state.get("proposal"))
    quant_dict = _to_dict(ctx.state.get("quant_analysis"))
    financial_dict = _to_dict(ctx.state.get("financial_analysis"))
    macro_dict = _to_dict(ctx.state.get("macro_analysis"))
    sentiment_dict = _to_dict(ctx.state.get("sentiment_analysis"))
    institutional_dict = _to_dict(ctx.state.get("institutional_analysis"))

    consensus_context = f"""
Current Proposal: {proposal_dict}
Quant Analysis: {quant_dict}
Financial Analysis: {financial_dict}
Macro Analysis: {macro_dict}
Sentiment Analysis: {sentiment_dict}
Institutional Flow Analysis: {institutional_dict}
Valuation Analysis: {revised_valuation}
Risk Analysis: {risk_dict}
"""
    consensus_check = await ctx.run_node(
        consensus_evaluator, node_input=consensus_context
    )
    if consensus_check and getattr(consensus_check, "revised_proposal", None):
        consensus_check.revised_proposal = _normalize_proposal_ticker(
            consensus_check.revised_proposal, ctx.state.get("market")
        )
    ctx.state["consensus"] = consensus_check

    yield Event(output=consensus_check)


@node
def finish_node(ctx: Context, node_input: ConsensusCheck):
    # Construct a beautiful markdown output for the Web Playground UI
    proposal = _to_dict(ctx.state.get("proposal"))
    quant = _to_dict(ctx.state.get("quant_analysis"))
    financial = _to_dict(ctx.state.get("financial_analysis"))
    macro = _to_dict(ctx.state.get("macro_analysis"))
    sentiment = _to_dict(ctx.state.get("sentiment_analysis"))
    institutional = _to_dict(ctx.state.get("institutional_analysis"))
    valuation = _to_dict(ctx.state.get("valuation_analysis"))
    revised_val = _to_dict(ctx.state.get("revised_valuation"))
    risk = _to_dict(ctx.state.get("risk_analysis"))

    markdown = f"""# 🏛️ Institutional Investment Committee Final Report

## 📢 Consensus Decision: **{node_input.final_decision}**

**Executive Summary & Rationale:**
{node_input.rationale}
"""

    curr = "₹" if ctx.state.get("market") == "IN" else "$"

    if ctx.state.get("debate_triggered") and revised_val:
        markdown += f"""
### ⚖️ Debate & Resolution (Chief Investment Officer Gate)
*A structural discrepancy was detected between Valuation and Risk profiles. Debate triggered:*
- **CRO (Risk Analyst) Concern:** Highlighted a risk rating of **{(risk or {}).get("overall_risk_rating")}** with tail risk metrics (VaR 95%: {(risk or {}).get("quantitative_risk", {}).get("value_at_risk_95")}%).
- **Valuation Expert Response:** Revised valuation status to **{revised_val.get("valuation_status")}** with blended fair value of **{curr}{revised_val.get("intrinsic_value", {}).get("blended_fair_value")}**.
"""

    markdown += f"""
### 📊 Agent Assessment Summary:
1. **Portfolio PM Action:** {(proposal or {}).get("action")} (Target: {curr}{(proposal or {}).get("price_target")})
2. **Quant Arb:** Cointegrated: {(quant or {}).get("statistical_arbitrage", {}).get("is_cointegrated")!s} (Spread Z: {(quant or {}).get("statistical_arbitrage", {}).get("spread_z_score")}, Sig: {(quant or {}).get("signal_validation", {}).get("statistical_significance")})
3. **Financial Solvency:** {(financial or {}).get("forensic_analysis", {}).get("bankruptcy_risk")} (Altman Z: {(financial or {}).get("forensic_analysis", {}).get("altman_z_score")})
4. **Macro Outlook:** {(macro or {}).get("macro_environment_status")}
5. **News Sentiment:** {(sentiment or {}).get("overall_sentiment_status")}
6. **Smart Money Flow:** {(institutional or {}).get("smart_money_sentiment")} (Insider Verdict: {(institutional or {}).get("insider_activity", {}).get("verdict")})
7. **Valuation Appraisal:** {(valuation or {}).get("valuation_status")} (Blended Intrinsic Value: {curr}{(valuation or {}).get("intrinsic_value", {}).get("blended_fair_value")})
8. **CRO Risk Rating:** {(risk or {}).get("overall_risk_rating")}
"""

    if node_input.revised_proposal:
        prop = node_input.revised_proposal
        markdown += f"""
### 🎯 Final Approved Proposal Details:
- **Ticker:** {prop.ticker}
- **Sector:** {prop.sector}
- **Proposed Action:** {prop.action}
- **Price Target:** {curr}{prop.price_target:.2f}
- **Rationale:** {prop.rationale}
"""
    else:
        markdown += "\n- No revised proposal was generated; consensus was reached on the initial thesis."

    yield Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=markdown)])
    )
    yield Event(output=markdown)


# ---------------------------------------------------------------------
# 6. Workflow Graph Configuration & Connection
# ---------------------------------------------------------------------

join_node = JoinNode(name="merge_analysis")

edges = [
    # Start goes into portfolio coordinator (unconditional)
    ("START", portfolio_manager),
    # Portfolio manager routes conditionally
    (
        portfolio_manager,
        {
            "continue": (
                quant_agent,
                financial_agent,
                valuation_agent,
                macro_agent,
                sentiment_agent,
                risk_agent,
                institutional_agent,
            ),
            "consensus_reached": finish_node,
            "trigger_debate": debate_node,
            "security_flagged": human_review,
        },
    ),
    # human_review routes conditionally based on human choice
    (
        human_review,
        {
            "continue": (
                quant_agent,
                financial_agent,
                valuation_agent,
                macro_agent,
                sentiment_agent,
                risk_agent,
                institutional_agent,
            ),
            "exit": finish_node,
        },
    ),
    # Join parallel outputs (unconditional)
    (
        (
            quant_agent,
            financial_agent,
            valuation_agent,
            macro_agent,
            sentiment_agent,
            risk_agent,
            institutional_agent,
        ),
        join_node,
    ),
    # Join node loops back to portfolio coordinator (unconditional)
    (join_node, portfolio_manager),
    # Debate node loops back to portfolio coordinator (unconditional)
    (debate_node, portfolio_manager),
]

workflow = Workflow(
    name="investment_comittee_workflow",
    edges=edges,
    state_schema=WorkflowState,
)

root_agent = workflow

# ---------------------------------------------------------------------
# 7. ADK App Container
# ---------------------------------------------------------------------

app = App(
    root_agent=workflow,
    name="app",
)
