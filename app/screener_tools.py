"""NSE Stock Screener — fetches live Nifty index constituents from NSE archives and
screens them using yfinance fundamentals with risk-profile-weighted scoring.

Data sources (all free, no API key required):
  - NSE India index CSV archives (updated daily):
      Nifty 500:       https://archives.nseindia.com/content/indices/ind_nifty500list.csv
      Nifty Microcap:  https://archives.nseindia.com/content/indices/ind_niftymicrocap250list.csv
      Nifty 50:        https://archives.nseindia.com/content/indices/ind_nifty50list.csv
  - yfinance for per-ticker fundamentals (P/E, P/B, ROE, debt/equity, 52w return)
"""

import io
import logging
import time
import urllib.request
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSE index CSV URLs (public, no auth required)
# ---------------------------------------------------------------------------
_NSE_INDEX_URLS: dict[str, str] = {
    "nifty500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "nifty50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "niftymidcap150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "niftysmallcap250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "niftymicrocap250": "https://archives.nseindia.com/content/indices/ind_niftymicrocap250list.csv",
    "nifty100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
}

# NSE industry label → yfinance sector keywords for matching
_SECTOR_ALIASES: dict[str, list[str]] = {
    "information technology": ["technology", "software", "it services", "computers"],
    "banking": ["bank", "financial services", "banking"],
    "financial services": ["financial", "nbfc", "insurance", "bank"],
    "pharmaceuticals": ["pharma", "drug", "healthcare", "biotech", "medicine"],
    "fmcg": ["consumer", "fmcg", "beverage", "food", "personal products"],
    "consumer goods": ["consumer", "fmcg", "beverage", "food"],
    "automobile": ["auto", "vehicle", "motor", "transport equipment"],
    "energy": ["energy", "oil", "gas", "petroleum", "power"],
    "metals": ["metal", "steel", "aluminium", "mining", "zinc"],
    "telecom": ["telecom", "communication", "wireless"],
    "cement": ["cement", "construction material"],
    "infrastructure": ["infrastructure", "construction", "real estate"],
    "chemicals": ["chemical", "specialty chemicals", "fertilizer", "pesticide"],
    "textiles": ["textile", "garment", "apparel"],
}

# Cache: (url, date_str) → DataFrame
_universe_cache: dict[str, tuple[str, pd.DataFrame]] = {}

_SCORING_WEIGHTS: dict[str, dict[str, float]] = {
    "conservative": {"pe_score": 0.35, "debt_score": 0.30, "roe_score": 0.20, "return_score": 0.10, "pb_score": 0.05},
    "moderate":     {"pe_score": 0.25, "debt_score": 0.20, "roe_score": 0.25, "return_score": 0.20, "pb_score": 0.10},
    "high":         {"pe_score": 0.10, "debt_score": 0.15, "roe_score": 0.35, "return_score": 0.35, "pb_score": 0.05},
    "aggressive":   {"pe_score": 0.10, "debt_score": 0.10, "roe_score": 0.35, "return_score": 0.40, "pb_score": 0.05},
    "decent":       {"pe_score": 0.20, "debt_score": 0.20, "roe_score": 0.30, "return_score": 0.25, "pb_score": 0.05},
}


def _fetch_nse_universe(index_name: str = "nifty500") -> pd.DataFrame:
    """Download and cache the NSE index constituent CSV for today."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"{index_name}:{today_str}"

    if cache_key in _universe_cache:
        return _universe_cache[cache_key][1]

    url = _NSE_INDEX_URLS.get(index_name.lower(), _NSE_INDEX_URLS["nifty500"])
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Referer": "https://www.nseindia.com/",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw))
        # Standardise column names (NSE sometimes has trailing spaces)
        df.columns = [c.strip() for c in df.columns]
        _universe_cache[cache_key] = (today_str, df)
        logger.info("Fetched %d companies from NSE %s index", len(df), index_name)
        return df
    except Exception as e:
        logger.warning("NSE CSV fetch failed (%s): %s — using empty universe", index_name, e)
        return pd.DataFrame()


def _match_sector(nse_industry: str, target_sector: str) -> bool:
    """Return True if the NSE industry column matches the requested sector filter."""
    if not target_sector or target_sector.lower() in ("all", "open-ended", "any", ""):
        return True
    nse_lower = nse_industry.lower()
    target_lower = target_sector.lower()
    # Direct substring match
    if target_lower in nse_lower or nse_lower in target_lower:
        return True
    # Alias match
    for canonical, aliases in _SECTOR_ALIASES.items():
        if target_lower in canonical or canonical in target_lower:
            if any(alias in nse_lower for alias in aliases):
                return True
    return False


def _safe_float(val: Any) -> float | None:
    """Coerce a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _score_ticker(info: dict, risk_profile: str) -> float:
    """Compute a composite score (0-100) for a ticker based on key ratios.

    Higher score = better buy candidate for the given risk profile.
    Individual dimension scores (0-1):
        pe_score    - lower P/E is better (capped at 80; negative P/E = 0)
        pb_score    - lower P/B is better (capped at 20)
        roe_score   - higher ROE is better (capped at 50%)
        debt_score  - lower D/E is better (capped at 5)
        return_score- higher 52w return is better (linear, capped at +100% / -50%)
    """
    weights = _SCORING_WEIGHTS.get(risk_profile.lower(), _SCORING_WEIGHTS["moderate"])

    pe = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
    pb = _safe_float(info.get("priceToBook"))
    roe = _safe_float(info.get("returnOnEquity"))  # yfinance gives as decimal, e.g. 0.18 = 18%
    de = _safe_float(info.get("debtToEquity"))     # yfinance: total debt / equity as %
    week52_low = _safe_float(info.get("fiftyTwoWeekLow"))
    current = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

    # PE score: lower is better, normalised to 0-1 (PE of 10 = 1.0, PE of 80+ = 0.0)
    pe_score = 0.0
    if pe and pe > 0:
        pe_score = max(0.0, min(1.0, (80 - pe) / 70))

    # PB score: lower is better (PB of 1 = 1.0, PB of 10+ = 0.0)
    pb_score = 0.0
    if pb and pb > 0:
        pb_score = max(0.0, min(1.0, (10 - pb) / 9))

    # ROE score: higher is better (ROE 30%+ = 1.0, 0% = 0.0)
    roe_score = 0.0
    if roe is not None:
        roe_pct = roe * 100 if abs(roe) < 1.0 else roe  # normalise decimal→%
        roe_score = max(0.0, min(1.0, roe_pct / 30))

    # Debt score: lower D/E is better (0 = 1.0, 500%+ = 0.0)
    debt_score = 0.5  # neutral default when unavailable
    if de is not None and de >= 0:
        # yfinance returns debtToEquity as %, e.g. 45.23 means 45.23%
        debt_score = max(0.0, min(1.0, (500 - de) / 500))

    # 1Y return score: uses 52w range as proxy
    return_score = 0.5  # neutral default
    if week52_low and week52_low > 0 and current:
        approx_return = (current - week52_low) / week52_low  # rough 1Y return from trough
        # Normalise: +100% = 1.0, -50% = 0.0
        return_score = max(0.0, min(1.0, (approx_return + 0.5) / 1.5))

    composite = (
        weights["pe_score"] * pe_score
        + weights["pb_score"] * pb_score
        + weights["roe_score"] * roe_score
        + weights["debt_score"] * debt_score
        + weights["return_score"] * return_score
    )
    return round(composite * 100, 2)


def screen_nse_stocks(
    sector: str = "",
    risk_profile: str = "moderate",
    max_results: int = 5,
    index: str = "nifty500",
) -> dict:
    """Screen NSE-listed stocks from the Nifty index universe for the best investment candidates.

    Fetches the live Nifty index constituent list from NSE archives (free, no API key),
    filters by sector, then batch-fetches key fundamentals via yfinance and ranks by
    a risk-profile-weighted composite score.

    Args:
        sector: Industry/sector to filter on, e.g. 'Information Technology', 'Banking',
                'Pharmaceuticals', 'FMCG', 'Automobile'. Pass '' or 'open-ended' for all sectors.
        risk_profile: Investor risk appetite — 'conservative', 'moderate', 'decent', 'high',
                      or 'aggressive'. Determines the scoring weight matrix.
        max_results: Number of top-ranked candidates to return (default 5).
        index: Nifty index to use as the universe:
               'nifty500' (default, ~97% of market cap),
               'nifty50' (blue chips only),
               'nifty100', 'niftymidcap150', 'niftysmallcap250', 'niftymicrocap250'.

    Returns:
        A dict with keys:
          - status: 'success' or 'error'
          - universe_size: number of companies fetched from NSE
          - sector_filter: the sector filter applied
          - risk_profile: the risk profile used
          - candidates: list of dicts, each containing ticker, company, industry,
                        pe_ratio, pb_ratio, roe_pct, debt_to_equity, composite_score
    """
    try:
        # Step 1: Fetch the universe from NSE
        df = _fetch_nse_universe(index)
        if df.empty:
            return {
                "status": "error",
                "message": f"Could not download NSE {index} constituent list. Network issue or NSE server unavailable.",
            }

        # Identify relevant columns (NSE CSV uses 'Symbol' and 'Industry')
        symbol_col = next((c for c in df.columns if "symbol" in c.lower()), None)
        industry_col = next(
            (c for c in df.columns if "industry" in c.lower() or "sector" in c.lower()), None
        )
        name_col = next((c for c in df.columns if "company" in c.lower() or "name" in c.lower()), None)

        if not symbol_col:
            return {"status": "error", "message": f"NSE CSV format unexpected. Columns: {list(df.columns)}"}

        # Step 2: Apply sector filter
        if industry_col and sector and sector.lower() not in ("", "all", "open-ended", "any"):
            mask = df[industry_col].fillna("").apply(lambda x: _match_sector(str(x), sector))
            filtered_df = df[mask]
            if filtered_df.empty:
                # Fall back to full universe if no sector match
                logger.warning("No NSE companies matched sector '%s' — using full universe", sector)
                filtered_df = df
        else:
            filtered_df = df

        universe_size = len(filtered_df)

        # Build ticker list (.NS suffix for NSE)
        raw_symbols = filtered_df[symbol_col].dropna().tolist()
        tickers = [f"{str(s).strip()}.NS" for s in raw_symbols if str(s).strip()]

        # Step 3: Batch-fetch fundamentals (process in chunks to avoid yfinance rate limits)
        # Cap at 150 tickers to stay within reasonable time bounds
        tickers = tickers[:150]

        scored: list[dict] = []
        chunk_size = 20
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            for ticker in chunk:
                try:
                    obj = yf.Ticker(ticker)
                    info = obj.info or {}
                    # Skip if no meaningful data returned
                    if not info.get("regularMarketPrice") and not info.get("currentPrice"):
                        continue
                    score = _score_ticker(info, risk_profile)

                    # Get company name from NSE CSV if available
                    symbol_bare = ticker.replace(".NS", "")
                    company_row = filtered_df[filtered_df[symbol_col].astype(str).str.strip() == symbol_bare]
                    company_name = (
                        str(company_row[name_col].iloc[0]).strip()
                        if name_col and not company_row.empty
                        else info.get("longName", ticker)
                    )
                    industry_label = (
                        str(company_row[industry_col].iloc[0]).strip()
                        if industry_col and not company_row.empty
                        else info.get("sector", "Unknown")
                    )

                    roe = _safe_float(info.get("returnOnEquity"))
                    de = _safe_float(info.get("debtToEquity"))
                    scored.append(
                        {
                            "ticker": ticker,
                            "company": company_name,
                            "industry": industry_label,
                            "market_cap_cr": round(
                                (_safe_float(info.get("marketCap")) or 0) / 1e7, 0
                            ),  # convert to Crores
                            "current_price_inr": _safe_float(
                                info.get("currentPrice") or info.get("regularMarketPrice")
                            ),
                            "pe_ratio": _safe_float(info.get("trailingPE") or info.get("forwardPE")),
                            "pb_ratio": _safe_float(info.get("priceToBook")),
                            "roe_pct": round(roe * 100, 1) if roe is not None else None,
                            "debt_to_equity": de,
                            "composite_score": score,
                        }
                    )
                except Exception as e:
                    logger.debug("Skipping %s: %s", ticker, e)
                    continue
            # Small pause between chunks to be polite to yfinance
            if i + chunk_size < len(tickers):
                time.sleep(0.3)

        if not scored:
            return {
                "status": "error",
                "message": "yfinance returned no valid data for any ticker in the filtered universe.",
            }

        # Step 4: Rank and return top N
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        top = scored[:max_results]

        return {
            "status": "success",
            "universe_size": universe_size,
            "tickers_screened": len(scored),
            "index_used": index,
            "sector_filter": sector or "all sectors",
            "risk_profile": risk_profile,
            "candidates": top,
        }

    except Exception as e:
        logger.exception("screen_nse_stocks failed")
        return {"status": "error", "message": f"Screener error: {e!s}"}
