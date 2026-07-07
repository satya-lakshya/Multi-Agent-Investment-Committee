import io
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_fred_macro_data(indicator_codes: list | None = None) -> dict:
    """Connects to the FRED database via public CSV links to pull live economic numbers.

    Args:
        indicator_codes: A list of FRED series IDs to fetch (e.g., ['DGS10', 'CPIAUCSL', 'FEDFUNDS']).
                         If None, defaults to ['DGS10', 'CPIAUCSL', 'FEDFUNDS'].

    Returns:
        A dictionary containing the parsed FRED data, including latest value, observation date,
        and trend over recent observations.
    """
    if not indicator_codes:
        indicator_codes = ["DGS10", "CPIAUCSL", "FEDFUNDS"]

    results = {}
    monthly_indicators = {"CPIAUCSL"}

    for code in indicator_codes:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
            # Fetch using pandas
            df = pd.read_csv(url)

            # Clean missing data (FRED uses '.' for missing values)
            df = df[df[code] != "."]
            df[code] = pd.to_numeric(df[code])

            if df.empty:
                results[code] = {
                    "status": "error",
                    "message": f"No valid data found for series '{code}'.",
                }
                continue

            latest_row = df.iloc[-1]
            latest_val = float(latest_row[code])
            latest_date = str(latest_row["observation_date"])

            # Determine historical window for trend comparison
            window = 6 if code in monthly_indicators else 30
            if len(df) >= window + 1:
                past_val = float(df.iloc[-(window + 1)][code])
            else:
                past_val = float(df.iloc[0][code])

            # Compute trend based on series type
            if code == "CPIAUCSL":
                # CPI Index: calculate percentage change
                pct_change = ((latest_val - past_val) / past_val) * 100
                if pct_change > 0.5:
                    trend = "RISING"
                elif pct_change < -0.5:
                    trend = "FALLING"
                else:
                    trend = "FLAT"
            else:
                # Interest rates / yields: calculate absolute difference in percentage points
                diff = latest_val - past_val
                if diff > 0.1:
                    trend = "RISING"
                elif diff < -0.1:
                    trend = "FALLING"
                else:
                    trend = "FLAT"

            results[code] = {
                "status": "success",
                "latest_value": latest_val,
                "observation_date": latest_date,
                "trend": trend,
            }
        except Exception as e:
            results[code] = {
                "status": "error",
                "message": f"Failed to fetch {code}: {e!s}",
            }

    return results


def fetch_india_macro_data() -> dict:
    """Fetches Indian macroeconomic indicators using the World Bank API and yfinance proxies.

    Covers:
    - India CPI Inflation (World Bank: NY.GDP.DEFL.KD.ZG proxy, FP.CPI.TOTL.ZG)
    - USD/INR exchange rate (yfinance: INRUSD=X)
    - Nifty 50 52-week trend (yfinance: ^NSEI) as a market regime indicator

    Returns:
        A dictionary with latest values and trends for each India macro indicator.
    """
    results = {}

    # 1. India CPI from World Bank API (annual, latest available)
    try:
        url = "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL.ZG?format=json&mrv=3"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            entries = data[1] if len(data) > 1 else []
            valid = [e for e in entries if e.get("value") is not None]
            if valid:
                latest = valid[0]
                prev = valid[1] if len(valid) > 1 else None
                cpi_val = float(latest["value"])
                trend = "FLAT"
                if prev and prev.get("value"):
                    diff = cpi_val - float(prev["value"])
                    trend = "RISING" if diff > 0.2 else ("FALLING" if diff < -0.2 else "FLAT")
                results["IN_CPI_INFLATION"] = {
                    "status": "success",
                    "latest_value": round(cpi_val, 2),
                    "observation_date": str(latest.get("date")),
                    "unit": "percent annual",
                    "trend": trend,
                }
    except Exception as e:
        results["IN_CPI_INFLATION"] = {"status": "error", "message": str(e)}

    # 2. USD/INR exchange rate via yfinance
    try:
        inr = yf.Ticker("INRUSD=X").history(period="3mo")
        if not inr.empty:
            latest_rate = float(inr["Close"].iloc[-1])
            prev_rate = float(inr["Close"].iloc[0])
            diff_pct = ((latest_rate - prev_rate) / prev_rate) * 100
            trend = "APPRECIATING" if diff_pct > 0.5 else ("DEPRECIATING" if diff_pct < -0.5 else "STABLE")
            results["INRUSD"] = {
                "status": "success",
                "latest_value": round(latest_rate, 4),
                "observation_date": str(datetime.now().date()),
                "unit": "INR per USD (inverted)",
                "trend": trend,
            }
    except Exception as e:
        results["INRUSD"] = {"status": "error", "message": str(e)}

    # 3. Nifty 50 trend as market regime proxy
    try:
        nsei = yf.Ticker("^NSEI").history(period="6mo")
        if not nsei.empty:
            start_p = float(nsei["Close"].iloc[0])
            end_p = float(nsei["Close"].iloc[-1])
            perf_pct = ((end_p - start_p) / start_p) * 100
            regime = "BULL" if perf_pct > 5 else ("BEAR" if perf_pct < -5 else "SIDEWAYS")
            results["NIFTY50_6M"] = {
                "status": "success",
                "latest_value": round(end_p, 2),
                "6m_return_pct": round(perf_pct, 2),
                "market_regime": regime,
            }
    except Exception as e:
        results["NIFTY50_6M"] = {"status": "error", "message": str(e)}

    return results


def analyze_sector_momentum(sector_ticker: str, benchmark: str = "") -> dict:
    """Compares sector performance vs a benchmark over a 6-month rolling window.

    Auto-detects the benchmark: Indian tickers and NSE indices (^CNXIT, ^NSEBANK, etc.)
    default to Nifty 50 (^NSEI); US tickers default to SPY.

    Indian NSE sector indices you can pass as sector_ticker:
        ^CNXIT  — Nifty IT
        ^NSEBANK — Nifty Bank
        ^CNXAUTO — Nifty Auto
        ^CNXPHARMA — Nifty Pharma
        ^CNXFMCG — Nifty FMCG
        ^CNXENERGY — Nifty Energy
        ^CNXMETAL — Nifty Metal

    Args:
        sector_ticker: The sector ETF or index ticker (e.g., XLK, ^CNXIT).
        benchmark: Benchmark ticker. Auto-detected if empty.

    Returns:
        A dictionary containing 6-month rolling returns, outperformance, and lifecycle stage.
    """
    # Auto-detect benchmark
    is_indian_sector = (
        sector_ticker.startswith("^CNX")
        or sector_ticker.startswith("^NSE")
        or sector_ticker.upper().endswith(".NS")
        or sector_ticker.upper().endswith(".BO")
    )
    if not benchmark:
        benchmark = "^NSEI" if is_indian_sector else "SPY"

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months

        # Fetch sector ETF data
        sector_obj = yf.Ticker(sector_ticker)
        df_sector = sector_obj.history(
            start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
        )

        # Fetch benchmark ETF data
        bench_obj = yf.Ticker(benchmark)
        df_bench = bench_obj.history(
            start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
        )

        if df_sector.empty or df_bench.empty:
            return {
                "status": "error",
                "message": f"Data not found for ticker '{sector_ticker}' or benchmark '{benchmark}'.",
            }

        # Calculate performance
        sec_start = float(df_sector["Close"].iloc[0])
        sec_end = float(df_sector["Close"].iloc[-1])
        sector_perf = ((sec_end - sec_start) / sec_start) * 100

        bench_start = float(df_bench["Close"].iloc[0])
        bench_end = float(df_bench["Close"].iloc[-1])
        bench_perf = ((bench_end - bench_start) / bench_start) * 100

        diff = sector_perf - bench_perf

        if diff > 1.0:
            relative = "OUTPERFORMING"
        elif diff < -1.0:
            relative = "UNDERPERFORMING"
        else:
            relative = "MATCHING"

        # Determine lifecycle stage
        if sector_perf > 5.0 and relative == "OUTPERFORMING":
            stage = "GROWTH"
        elif sector_perf > 0.0:
            stage = "MATURE"
        elif sector_perf <= 0.0 and relative == "UNDERPERFORMING":
            stage = "DECLINE"
        else:
            stage = "ACCELERATING"

        return {
            "status": "success",
            "sector_ticker": sector_ticker,
            "sector_etf_performance_6m": round(sector_perf, 2),
            "benchmark_ticker": benchmark,
            "benchmark_performance_6m": round(bench_perf, 2),
            "relative_to_sp500": relative,
            "lifecycle_stage": stage,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to analyze sector momentum: {e!s}",
        }


def fetch_geopolitical_risk_signals(keywords: list) -> dict:
    """Queries news channels using specific keywords to search for geopolitical and regulatory risks.

    Supports Alpha Vantage News Sentiment API if an API key is present in environment,
    otherwise falls back to a Google News RSS parsing engine.

    Args:
        keywords: A list of sector/macro keywords to search for (e.g., ['tariffs', 'semiconductor export restrictions']).

    Returns:
        A dictionary containing fetched articles, titles, sources, and links.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if api_key:
        try:
            # Query Alpha Vantage News Sentiment
            query_str = ",".join(keywords)
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={urllib.parse.quote(query_str)}&apikey={api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                data = pd.read_json(io.StringIO(resp.read().decode()))
                # Parse feed
                feed = data.get("feed", [])
                articles = []
                for item in feed[:10]:
                    articles.append(
                        {
                            "title": item.get("title"),
                            "summary": item.get("summary"),
                            "source": item.get("source"),
                            "url": item.get("url"),
                            "time_published": item.get("time_published"),
                        }
                    )
                return {
                    "status": "success",
                    "source": "alpha_vantage",
                    "articles": articles,
                }
        except Exception:
            # Fall back to RSS if Alpha Vantage fails
            pass

    # Google News RSS Fallback (keyless)
    try:
        # Construct RSS search query
        query_parts = []
        for kw in keywords:
            if " " in kw:
                query_parts.append(f'"{kw}"')
            else:
                query_parts.append(kw)
        query_str = " OR ".join(query_parts)

        rss_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query_str)}&hl=en-US&gl=US&ceid=US:en"

        req = urllib.request.Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)

            articles = []
            for item in root.findall(".//item")[:10]:
                title = (
                    item.find("title").text if item.find("title") is not None else ""
                )
                link = item.find("link").text if item.find("link") is not None else ""
                pub_date = (
                    item.find("pubDate").text
                    if item.find("pubDate") is not None
                    else ""
                )
                source = (
                    item.find("source").text if item.find("source") is not None else ""
                )

                articles.append(
                    {
                        "title": title,
                        "url": link,
                        "pub_date": pub_date,
                        "source": source,
                    }
                )

            return {
                "status": "success",
                "source": "google_news_rss",
                "articles": articles,
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch geopolitical news: {e!s}",
        }
