import numpy as np
import yfinance as yf

# In-memory cache for peer info queries to speed up execution
_valuation_cache = {}


def get_multiple(info, keys) -> float | None:
    """Safely extracts a numeric multiple from a yfinance info dict."""
    for key in keys:
        val = info.get(key)
        if val is not None:
            try:
                f_val = float(val)
                if not np.isnan(f_val) and f_val > 0:
                    return f_val
            except (ValueError, TypeError):
                continue
    return None


def calculate_absolute_dcf(
    fcf_array: list,
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float | None = None,
) -> dict:
    """Programmatically calculates the Discounted Cash Flow (DCF) intrinsic value.

    Formula:
    PV = sum(FCF_t / (1 + WACC)^t) + TV / (1 + WACC)^n
    TV = FCF_n * (1 + g_terminal) / (WACC - g_terminal)

    Args:
        fcf_array: A list of future projected or historical Free Cash Flows (floats).
        wacc: Weighted Average Cost of Capital (as percentage, e.g. 8.5, or decimal, e.g. 0.085).
        terminal_growth: Terminal growth rate (as percentage, e.g. 3.0, or decimal, e.g. 0.03).
        shares_outstanding: Current shares outstanding to compute per-share value.

    Returns:
        A dictionary containing total DCF implied value, per-share implied value, WACC and growth rates.
    """
    if not fcf_array:
        return {"status": "error", "message": "fcf_array must not be empty"}

    # Normalize WACC and Terminal Growth to decimals
    wacc_dec = wacc / 100.0 if wacc > 1.0 else wacc
    growth_dec = terminal_growth / 100.0 if terminal_growth > 1.0 else terminal_growth

    # Handle edge case where WACC <= terminal growth rate to prevent infinite/negative TV
    if wacc_dec <= growth_dec:
        wacc_dec = growth_dec + 0.02  # enforce a 2% spread minimum

    n = len(fcf_array)

    # 1. Discount cash flows
    pv_cashflows = 0.0
    for t, fcf in enumerate(fcf_array):
        pv_cashflows += float(fcf) / ((1.0 + wacc_dec) ** (t + 1))

    # 2. Gordon Growth Terminal Value
    last_fcf = float(fcf_array[-1])
    tv = (last_fcf * (1.0 + growth_dec)) / (wacc_dec - growth_dec)
    pv_tv = tv / ((1.0 + wacc_dec) ** n)

    total_dcf_value = pv_cashflows + pv_tv

    # 3. Calculate per-share value if shares outstanding is provided
    per_share_value = total_dcf_value
    if shares_outstanding and shares_outstanding > 0:
        per_share_value = total_dcf_value / shares_outstanding

    return {
        "status": "success",
        "total_dcf_value": total_dcf_value,
        "per_share_value": per_share_value,
        "wacc_applied": wacc_dec,
        "terminal_growth_applied": growth_dec,
    }


def calculate_graham_intrinsic_value(
    eps: float, growth_rate: float, corporate_bond_yield: float
) -> dict:
    """Executes Benjamin Graham's revised intrinsic value formula from The Intelligent Investor.

    Formula:
    V = (EPS * (8.5 + 2 * g) * 4.4) / Y

    Args:
        eps: Earnings Per Share (TTM or normalized).
        growth_rate: Expected growth rate (as percentage, e.g. 5.0, or decimal, e.g. 0.05).
        corporate_bond_yield: Current yield of AAA corporate bonds (as percentage, e.g. 4.4, or decimal, e.g. 0.044).

    Returns:
        A dictionary containing the calculated Graham intrinsic value.
    """
    # Normalize growth rate and corporate bond yield to percentage format (e.g. 5% -> 5.0)
    g_val = growth_rate * 100.0 if 0.0 < growth_rate < 1.0 else growth_rate
    y_val = (
        corporate_bond_yield * 100.0
        if 0.0 < corporate_bond_yield < 1.0
        else corporate_bond_yield
    )

    # Fallback to standard 4.4% if bond yield is invalid
    if y_val <= 0:
        y_val = 4.4

    v = (eps * (8.5 + 2.0 * g_val) * 4.4) / y_val

    return {
        "status": "success",
        "graham_intrinsic_value": max(0.0, v),
        "eps": eps,
        "growth_rate_applied": g_val,
        "bond_yield_applied": y_val,
    }


def calculate_relative_multiples(ticker: str, peer_group: list) -> dict:
    """Calculates standard valuation factors for the target stock and its immediate competitors
    and ranks them using Jansen's factor-ranking approach.

    Args:
        ticker: The target stock ticker symbol.
        peer_group: List of immediate peer ticker symbols.

    Returns:
        A dictionary of the target stock's multiples and standard deviation Z-score rankings.
    """
    # ---------------------------------------------------------------------------
    # Smart peer group resolution: auto-detect Indian (.NS/.BO) vs US tickers
    # For Indian tickers: query yfinance for the sector, then fetch same-sector
    # Nifty 50 blue-chip peers dynamically from screener_tools.
    # ---------------------------------------------------------------------------
    _US_DEFAULT_PEERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    # Nifty 50 diversified blue-chips — last-resort fallback only
    _NIFTY50_BLUECHIPS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    ]

    is_indian = ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")

    if not peer_group:
        if is_indian:
            try:
                from app.screener_tools import (
                    screen_nse_stocks,  # avoid circular at module level
                )

                info = yf.Ticker(ticker).info
                detected_sector = info.get("sector", "") or ""
                # Map yfinance sector to NSE industry label
                _YF_TO_NSE: dict[str, str] = {
                    "Technology": "Information Technology",
                    "Financial Services": "Financial Services",
                    "Healthcare": "Pharmaceuticals",
                    "Consumer Defensive": "FMCG",
                    "Consumer Cyclical": "Automobile",
                    "Energy": "Energy",
                    "Basic Materials": "Metals",
                    "Communication Services": "Telecom",
                    "Industrials": "Infrastructure",
                    "Utilities": "Energy",
                }
                nse_sector = _YF_TO_NSE.get(detected_sector, detected_sector)
                result = screen_nse_stocks(
                    sector=nse_sector or "",
                    risk_profile="moderate",
                    max_results=6,
                    index="nifty50",  # use Nifty 50 for peers — guaranteed liquidity
                )
                if result.get("status") == "success" and result.get("candidates"):
                    peer_group = [
                        c["ticker"]
                        for c in result["candidates"]
                        if c["ticker"].upper() != ticker.upper()
                    ][:5]
                else:
                    peer_group = _NIFTY50_BLUECHIPS
            except Exception:
                peer_group = _NIFTY50_BLUECHIPS
        else:
            peer_group = _US_DEFAULT_PEERS


    all_tickers = list({ticker, *peer_group})
    multiples_data = {}

    for t in all_tickers:
        if t in _valuation_cache:
            multiples_data[t] = _valuation_cache[t]
            continue
        try:
            t_obj = yf.Ticker(t)
            info = t_obj.info
            multiples_data[t] = {
                "pe": get_multiple(info, ["trailingPE", "forwardPE"]),
                "peg": get_multiple(info, ["trailingPegRatio", "pegRatio"]),
                "ev_ebitda": get_multiple(info, ["enterpriseToEbitda"]),
                "ps": get_multiple(info, ["priceToSalesTrailing12Months"]),
                "pb": get_multiple(info, ["priceToBook"]),
            }
            _valuation_cache[t] = multiples_data[t]
        except Exception:
            multiples_data[t] = {
                "pe": None,
                "peg": None,
                "ev_ebitda": None,
                "ps": None,
                "pb": None,
            }

    target = multiples_data.get(ticker)
    if not target:
        return {
            "status": "error",
            "message": f"Unable to fetch info for target ticker '{ticker}'",
        }

    # Compute Z-scores for each multiple
    metrics = ["pe", "peg", "ev_ebitda", "ps", "pb"]
    z_scores = []

    for metric in metrics:
        vals = [
            multiples_data[t][metric]
            for t in all_tickers
            if multiples_data[t][metric] is not None
        ]
        if len(vals) >= 2 and target[metric] is not None:
            mean = np.mean(vals)
            std = np.std(vals)
            if std > 0:
                z = (target[metric] - mean) / std
                z_scores.append(z)

    # Average the Z-scores to get a blended relative value score
    if z_scores:
        blended_z = np.mean(z_scores)
    else:
        blended_z = 0.0

    # Map Z-score to relative pricing percentiles
    # Lower Z-score = cheaper stock relative to peers
    if blended_z <= -1.5:
        relative_val = "CHEAPER_THAN_95_PERCENT_OF_PEERS"
    elif blended_z <= -1.0:
        relative_val = "CHEAPER_THAN_85_PERCENT_OF_PEERS"
    elif blended_z <= -0.5:
        relative_val = "CHEAPER_THAN_70_PERCENT_OF_PEERS"
    elif blended_z <= 0.0:
        relative_val = "CHEAPER_THAN_55_PERCENT_OF_PEERS"
    elif blended_z <= 0.5:
        relative_val = "EXPENSIVE_THAN_55_PERCENT_OF_PEERS"
    elif blended_z <= 1.0:
        relative_val = "EXPENSIVE_THAN_70_PERCENT_OF_PEERS"
    elif blended_z <= 1.5:
        relative_val = "EXPENSIVE_THAN_85_PERCENT_OF_PEERS"
    else:
        relative_val = "EXPENSIVE_THAN_95_PERCENT_OF_PEERS"

    return {
        "status": "success",
        "pe_ratio": target.get("pe"),
        "peg_ratio": target.get("peg"),
        "ev_to_ebitda": target.get("ev_ebitda"),
        "price_to_sales": target.get("ps"),
        "price_to_book": target.get("pb"),
        "blended_z_score": float(blended_z),
        "sector_relative_valuation": relative_val,
    }


def calculate_margin_of_safety(
    current_price: float, blended_intrinsic_value: float
) -> dict:
    """Calculates the percentage difference between the asset's true worth and its current price.

    Args:
        current_price: The asset's current market price.
        blended_intrinsic_value: The estimated fair value.

    Returns:
        A dictionary containing the Margin of Safety percentage.
    """
    if blended_intrinsic_value <= 0:
        return {
            "status": "success",
            "margin_of_safety_percentage": 0.0,
            "message": "No Margin of Safety: Intrinsic value is non-positive.",
        }

    mos = (blended_intrinsic_value - current_price) / blended_intrinsic_value
    mos_pct = round(float(mos * 100.0), 2)

    return {"status": "success", "margin_of_safety_percentage": mos_pct}
