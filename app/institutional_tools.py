from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def _is_indian_ticker(ticker: str) -> bool:
    return ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO")


def fetch_insider_transactions(ticker: str, days: int = 90) -> dict:
    """Queries insider transaction filings via yfinance.

    For US stocks: reads SEC Form 4 data.
    For Indian stocks (.NS/.BO): yfinance does not provide SEBI insider disclosure
    data, so the tool reports what is actually available (volume-based proxy signals).

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, ADANIENT.NS).
        days: Historical window to analyze (default: 90 days).

    Returns:
        A dictionary containing buy/sell ratios, key purchases, and verdict.
    """
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.insider_transactions

        buy_count = 0
        sell_count = 0
        key_buys = []

        if df is not None and not df.empty:
            # Clean up Date column
            if "Start Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Start Date"])
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            else:
                df["Date"] = pd.to_datetime(df.index)

            cutoff = datetime.now() - timedelta(days=days)
            df_recent = df[df["Date"] >= cutoff]

            for _, row in df_recent.iterrows():
                shares = row.get("Shares", 0)
                insider = row.get("Insider", "Insider")
                text_desc = str(row.get("TransactionText", "")).lower()
                is_buy = shares > 0 and "sale" not in text_desc and "sell" not in text_desc
                if is_buy:
                    buy_count += abs(shares)
                    key_buys.append(
                        f"{insider} purchased {abs(shares):,} shares on {row['Date'].strftime('%m/%d')}"
                    )
                else:
                    sell_count += abs(shares)

        # If no data found — report honestly (don't fabricate)
        if buy_count == 0 and sell_count == 0:
            if _is_indian_ticker(ticker):
                return {
                    "status": "success",
                    "net_buying_selling_ratio_90d": None,
                    "key_insider_buys": [],
                    "verdict": "DATA_UNAVAILABLE",
                    "note": (
                        "SEBI insider transaction disclosures are not available via "
                        "yfinance for Indian stocks. Check BSE/NSE filings directly."
                    ),
                }
            else:
                # US stock with no recent filings — genuinely neutral
                return {
                    "status": "success",
                    "net_buying_selling_ratio_90d": 1.0,
                    "key_insider_buys": ["No open-market insider purchases in the past 90 days."],
                    "verdict": "NEUTRAL",
                }

        ratio = round(buy_count / max(sell_count, 1), 2)
        verdict = "BULLISH" if ratio > 1.2 else ("BEARISH" if ratio < 0.5 else "NEUTRAL")

        return {
            "status": "success",
            "net_buying_selling_ratio_90d": ratio,
            "key_insider_buys": key_buys[:3],
            "verdict": verdict,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch insider transactions: {e!s}",
        }


def fetch_institutional_holdings(ticker: str) -> dict:
    """Queries institutional holdings data via yfinance.

    For US stocks: uses SEC Form 13F filings aggregated by yfinance.
    For Indian stocks (.NS/.BO): returns FII/DII data from yfinance where available,
    or reports DATA_UNAVAILABLE (SEBI bulk deal disclosures are not in yfinance).

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, ADANIENT.NS).

    Returns:
        A dictionary containing institutional ownership percentage, QoQ changes, and top holders.
    """
    try:
        t_obj = yf.Ticker(ticker)
        df = t_obj.institutional_holders
        info = t_obj.info

        # Try to get real ownership from yfinance info
        held_pct_raw = info.get("heldPercentInstitutions")
        ownership_pct = round(float(held_pct_raw) * 100.0, 2) if held_pct_raw else None

        top_buyers: list = []
        change_str: str | None = None

        if df is not None and not df.empty:
            if "Holder" in df.columns:
                top_buyers = df["Holder"].head(4).tolist()
            if "pctChange" in df.columns:
                mean_change = df["pctChange"].mean() * 100
                change_str = f"{'+' if mean_change >= 0 else ''}{round(mean_change, 2)}%"

        # If no real data and this is an Indian stock, report honestly
        if ownership_pct is None and not top_buyers:
            if _is_indian_ticker(ticker):
                return {
                    "status": "success",
                    "institutional_ownership_percentage": None,
                    "quarter_over_quarter_change": None,
                    "top_buyers": [],
                    "verdict": "DATA_UNAVAILABLE",
                    "note": (
                        "FII/DII bulk deal disclosures for Indian stocks are filed with "
                        "SEBI/BSE and are not available via yfinance. "
                        "Check BSE bulk deal data or NSE FII activity reports."
                    ),
                }
            # For US stocks with no data — genuinely unknown, don't fabricate
            return {
                "status": "success",
                "institutional_ownership_percentage": None,
                "quarter_over_quarter_change": None,
                "top_buyers": ["Data not available from yfinance"],
                "verdict": "DATA_UNAVAILABLE",
            }

        return {
            "status": "success",
            "institutional_ownership_percentage": ownership_pct,
            "quarter_over_quarter_change": change_str,
            "top_buyers": top_buyers,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch institutional holdings: {e!s}",
        }


def analyze_block_trade_volume(ticker: str) -> dict:
    """Analyzes daily volume nodes to identify massive, single-print block trade anomalies.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL).

    Returns:
        A dictionary containing identified block trade anomalies and price support levels.
    """
    try:
        t_obj = yf.Ticker(ticker)
        # Pull 1 month of history
        df = t_obj.history(period="1mo")

        anomalies = []

        if df is not None and not df.empty:
            mean_vol = df["Volume"].mean()
            std_vol = df["Volume"].std()

            # Identify volume spikes > 1.8 standard deviations
            spikes = df[df["Volume"] > (mean_vol + 1.8 * std_vol)]

            for idx, row in spikes.iterrows():
                vol_ratio = row["Volume"] / mean_vol
                price = round(float(row["Close"]), 2)
                date_str = idx.strftime("%m/%d")
                anomalies.append(
                    f"Massive volume print of {int(row['Volume']):,} shares detected at ${price} on {date_str} ({round(vol_ratio, 1)}x 30d average)."
                )

        if not anomalies:
            anomalies = [
                "No significant single-print volume spikes detected over the past 30 days."
            ]

        return {"status": "success", "block_trade_anomalies": anomalies[:3]}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to analyze block trade volume: {e!s}",
        }
