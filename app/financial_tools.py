import pandas as pd
import yfinance as yf

# Fast in-memory cache to avoid duplicate yfinance requests
_filings_cache = {}


def get_val(df, keys, col) -> float:
    """Retrieves a numeric value from df for one of the keys in the specified column `col`.
    Handles case-insensitivity and strips whitespace.
    """
    if df is None or df.empty:
        return 0.0
    df_index_lower = [str(idx).strip().lower() for idx in df.index]

    for key in keys:
        key_clean = str(key).strip().lower()
        if key_clean in df_index_lower:
            idx_pos = df_index_lower.index(key_clean)
            val = df.iloc[idx_pos].loc[col]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            try:
                f_val = float(val)
                return f_val if not pd.isna(f_val) else 0.0
            except (ValueError, TypeError):
                return 0.0
    return 0.0


def fetch_standardized_filings(ticker: str, periods: int) -> dict:
    """Pulls historical financial statements (Income Statement, Balance Sheet, and Cash Flow)
    using yfinance and standardizes the key accounting line items.

    Args:
        ticker: The stock ticker symbol (e.g. AAPL, MSFT).
        periods: The number of years to pull (e.g. 5).

    Returns:
        A dictionary containing the parsed financial data and current price details.
    """
    cache_key = (ticker, periods)
    if cache_key in _filings_cache:
        return _filings_cache[cache_key]

    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info

        # Get current price
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("navPrice")
        )
        if not current_price:
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                current_price = float(hist["Close"].iloc[-1])
            else:
                current_price = 100.0  # fallback

        # Get current shares outstanding
        shares_outstanding = info.get("sharesOutstanding") or info.get(
            "impliedSharesOutstanding"
        )

        # Get statements
        financials = ticker_obj.financials
        balance_sheet = ticker_obj.balance_sheet
        cashflow = ticker_obj.cashflow

        if financials.empty or balance_sheet.empty or cashflow.empty:
            res = {
                "status": "error",
                "message": f"Unable to fetch complete financial statements for ticker '{ticker}'.",
            }
            _filings_cache[cache_key] = res
            return res

        # Standardize periods: align columns (dates) in ascending order (oldest to newest)
        cols = sorted(financials.columns)
        # Limit to the requested periods count
        cols = cols[-periods:]

        parsed_periods = []
        for col in cols:
            period_str = (
                col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
            )

            period_data = {
                "date": period_str,
                "revenue": get_val(
                    financials, ["Total Revenue", "Operating Revenue", "Revenue"], col
                ),
                "gross_profit": get_val(financials, ["Gross Profit"], col),
                "cost_of_revenue": get_val(
                    financials, ["Cost Of Revenue", "Reconciled Cost Of Revenue"], col
                ),
                "operating_income": get_val(
                    financials,
                    ["Operating Income", "Total Operating Income As Reported"],
                    col,
                ),
                "net_income": get_val(
                    financials,
                    [
                        "Net Income",
                        "Net Income Common Stockholders",
                        "Net Income Including Noncontrolling Interests",
                    ],
                    col,
                ),
                "ebit": get_val(financials, ["EBIT"], col),
                "sga": get_val(
                    financials,
                    [
                        "Selling General And Administration",
                        "Selling General & Administration",
                        "Total SGA",
                    ],
                    col,
                ),
                "depreciation": get_val(
                    financials,
                    [
                        "Depreciation And Amortization",
                        "Depreciation Amortization Depletion",
                        "Reconciled Depreciation",
                    ],
                    col,
                )
                or get_val(
                    cashflow,
                    [
                        "Depreciation And Amortization",
                        "Depreciation Amortization Depletion",
                    ],
                    col,
                ),
                "total_assets": get_val(balance_sheet, ["Total Assets"], col),
                "current_assets": get_val(balance_sheet, ["Current Assets"], col),
                "total_liabilities": get_val(
                    balance_sheet,
                    [
                        "Total Liabilities Net Minority Interest",
                        "Total Liabilities",
                        "Liabilities",
                    ],
                    col,
                ),
                "current_liabilities": get_val(
                    balance_sheet, ["Current Liabilities"], col
                ),
                "retained_earnings": get_val(balance_sheet, ["Retained Earnings"], col),
                "common_equity": get_val(
                    balance_sheet, ["Common Stock Equity", "Stockholders Equity"], col
                ),
                "long_term_debt": get_val(
                    balance_sheet,
                    ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
                    col,
                ),
                "total_debt": get_val(balance_sheet, ["Total Debt"], col),
                "inventory": get_val(balance_sheet, ["Inventory"], col),
                "receivables": get_val(
                    balance_sheet, ["Receivables", "Accounts Receivable"], col
                ),
                "net_ppe": get_val(
                    balance_sheet, ["Net PPE", "Gross PPE", "Properties"], col
                ),
                "operating_cash_flow": get_val(
                    cashflow,
                    [
                        "Operating Cash Flow",
                        "Cash Flow From Continuing Operating Activities",
                        "Cash Flow From Operating Activities",
                    ],
                    col,
                ),
                "capex": get_val(
                    cashflow,
                    [
                        "Capital Expenditure",
                        "Purchase Of PPE",
                        "Net PPE Purchase And Sale",
                    ],
                    col,
                ),
                "dividends_paid": get_val(
                    cashflow,
                    [
                        "Cash Dividends Paid",
                        "Common Stock Dividend Paid",
                        "Dividends Paid",
                    ],
                    col,
                ),
                "stock_issuance": get_val(
                    cashflow,
                    [
                        "Common Stock Issuance",
                        "Issuance Of Capital Stock",
                        "Net Common Stock Issuance",
                    ],
                    col,
                ),
                "shares_outstanding": get_val(
                    balance_sheet, ["Ordinary Shares Number", "Share Issued"], col
                ),
            }

            # Derived fields
            period_data["working_capital"] = (
                period_data["current_assets"] - period_data["current_liabilities"]
            )
            period_data["free_cash_flow"] = period_data["operating_cash_flow"] - abs(
                period_data["capex"]
            )

            parsed_periods.append(period_data)

        res = {
            "status": "success",
            "ticker": ticker,
            "current_price": current_price,
            "shares_outstanding_current": shares_outstanding,
            "periods": parsed_periods,
            "sector": info.get("sector", "unknown"),
        }
        _filings_cache[cache_key] = res
        return res
    except Exception as e:
        res = {
            "status": "error",
            "message": f"Failed to fetch filings for '{ticker}': {e!s}",
        }
        _filings_cache[cache_key] = res
        return res


def compute_forensic_scores(ticker: str, periods: int) -> dict:
    """Calculates Altman Z-Score (bankruptcy risk), Piotroski F-Score (efficiency), and Beneish M-Score (earnings manipulation).

    Args:
        ticker: The stock ticker symbol (e.g. AAPL).
        periods: The number of years of history to calculate metrics over (typically 5).

    Returns:
        A dictionary containing the Altman Z-Score, Piotroski F-Score, and Beneish M-Score.
    """
    try:
        financial_data = fetch_standardized_filings(ticker, periods)
        if financial_data.get("status") == "error":
            return financial_data

        periods_list = financial_data.get("periods", [])
        if not periods_list:
            return {
                "status": "error",
                "message": "No financial periods available to compute scores.",
            }

        latest = periods_list[-1]

        # Altman Z-Score inputs
        ta = latest.get("total_assets", 0.0)
        wc = latest.get("working_capital", 0.0)
        re_val = latest.get("retained_earnings", 0.0)
        ebit = latest.get("ebit", 0.0)
        tl = latest.get("total_liabilities", 0.0)
        sales = latest.get("revenue", 0.0)

        shares = (
            latest.get("shares_outstanding")
            or financial_data.get("shares_outstanding_current")
            or 0.0
        )
        price = financial_data.get("current_price", 0.0)
        mve = shares * price if shares and price else latest.get("common_equity", 0.0)

        sector = financial_data.get("sector", "unknown").lower()

        # Altman Z-Score calculation
        if ta > 0 and tl > 0:
            x1 = wc / ta
            x2 = re_val / ta
            x3 = ebit / ta
            x4 = mve / tl
            x5 = sales / ta

            is_manufacturing = any(
                m in sector
                for m in [
                    "manufactur",
                    "industrial",
                    "material",
                    "technology",
                    "hardware",
                ]
            )

            if is_manufacturing:
                z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
                if z_score > 2.99:
                    bankruptcy_risk = "LOW"
                elif z_score >= 1.81:
                    bankruptcy_risk = "GREY"
                else:
                    bankruptcy_risk = "HIGH"
            else:
                z_score = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
                if z_score > 2.90:
                    bankruptcy_risk = "LOW"
                elif z_score >= 1.23:
                    bankruptcy_risk = "GREY"
                else:
                    bankruptcy_risk = "HIGH"
        else:
            z_score = None
            bankruptcy_risk = "UNKNOWN"

        f_score_val = None
        m_score = None
        manipulation_flag = False

        if len(periods_list) >= 2:
            prev = periods_list[-2]
            ta_prev = prev.get("total_assets", 0.0)
            sales_prev = prev.get("revenue", 0.0)

            # --- Piotroski F-Score Calculations ---
            f_score = 0

            # 1. ROA > 0
            roa_curr = latest.get("net_income", 0.0) / ta if ta > 0 else 0.0
            if roa_curr > 0:
                f_score += 1

            # 2. CFO > 0
            cfo_curr = latest.get("operating_cash_flow", 0.0)
            if cfo_curr > 0:
                f_score += 1

            # 3. Change in ROA > 0
            roa_prev = prev.get("net_income", 0.0) / ta_prev if ta_prev > 0 else 0.0
            if roa_curr > roa_prev:
                f_score += 1

            # 4. Accruals: CFO > Net Income
            if cfo_curr > latest.get("net_income", 0.0):
                f_score += 1

            # 5. Change in Leverage (Debt/Assets decreased)
            lev_curr = latest.get("long_term_debt", 0.0) / ta if ta > 0 else 0.0
            lev_prev = prev.get("long_term_debt", 0.0) / ta_prev if ta_prev > 0 else 0.0
            if lev_curr < lev_prev:
                f_score += 1

            # 6. Change in Current Ratio (CR increased)
            cr_curr = (
                latest.get("current_assets", 0.0)
                / latest.get("current_liabilities", 1.0)
                if latest.get("current_liabilities", 0.0) > 0
                else 0.0
            )
            cr_prev = (
                prev.get("current_assets", 0.0) / prev.get("current_liabilities", 1.0)
                if prev.get("current_liabilities", 0.0) > 0
                else 0.0
            )
            if cr_curr > cr_prev:
                f_score += 1

            # 7. No Share Issuance (shares outstanding didn't increase)
            shares_curr = latest.get("shares_outstanding") or 0.0
            shares_prev = prev.get("shares_outstanding") or 0.0
            if shares_curr <= shares_prev:
                f_score += 1

            # 8. Change in Gross Margin (GM increased)
            gm_curr = latest.get("gross_profit", 0.0) / sales if sales > 0 else 0.0
            gm_prev = (
                prev.get("gross_profit", 0.0) / sales_prev if sales_prev > 0 else 0.0
            )
            if gm_curr > gm_prev:
                f_score += 1

            # 9. Change in Asset Turnover (AT increased)
            at_curr = sales / ta if ta > 0 else 0.0
            at_prev = sales_prev / ta_prev if ta_prev > 0 else 0.0
            if at_curr > at_prev:
                f_score += 1

            f_score_val = f_score

            # --- Beneish M-Score Calculations ---
            rec_curr = latest.get("receivables", 0.0)
            rec_prev = prev.get("receivables", 0.0)
            dsri_curr = rec_curr / sales if sales > 0 else 0.0
            dsri_prev = rec_prev / sales_prev if sales_prev > 0 else 0.0
            dsri = dsri_curr / dsri_prev if dsri_prev > 0 else 1.0

            gmi = gm_prev / gm_curr if gm_curr > 0 else 1.0

            aq_curr = (
                1.0
                - (
                    (latest.get("current_assets", 0.0) + latest.get("net_ppe", 0.0))
                    / ta
                )
                if ta > 0
                else 0.0
            )
            aq_prev = (
                1.0
                - (
                    (prev.get("current_assets", 0.0) + prev.get("net_ppe", 0.0))
                    / ta_prev
                )
                if ta_prev > 0
                else 0.0
            )
            aqi = aq_curr / aq_prev if aq_prev > 0 else 1.0

            sgi = sales / sales_prev if sales_prev > 0 else 1.0

            depr_curr = latest.get("depreciation", 0.0)
            depr_prev = prev.get("depreciation", 0.0)
            ppe_curr = latest.get("net_ppe", 0.0)
            ppe_prev = prev.get("net_ppe", 0.0)
            depr_rate_curr = (
                depr_curr / (ppe_curr + depr_curr)
                if (ppe_curr + depr_curr) > 0
                else 0.0
            )
            depr_rate_prev = (
                depr_prev / (ppe_prev + depr_prev)
                if (ppe_prev + depr_prev) > 0
                else 0.0
            )
            depi = depr_rate_prev / depr_rate_curr if depr_rate_curr > 0 else 1.0

            sgai_curr = latest.get("sga", 0.0) / sales if sales > 0 else 0.0
            sgai_prev = prev.get("sga", 0.0) / sales_prev if sales_prev > 0 else 0.0
            sgai = sgai_curr / sgai_prev if sgai_prev > 0 else 1.0

            lev_curr_m = (
                (
                    latest.get("current_liabilities", 0.0)
                    + latest.get("long_term_debt", 0.0)
                )
                / ta
                if ta > 0
                else 0.0
            )
            lev_prev_m = (
                (prev.get("current_liabilities", 0.0) + prev.get("long_term_debt", 0.0))
                / ta_prev
                if ta_prev > 0
                else 0.0
            )
            lvgi = lev_curr_m / lev_prev_m if lev_prev_m > 0 else 1.0

            tata = (latest.get("net_income", 0.0) - cfo_curr) / ta if ta > 0 else 0.0

            dsri = min(max(dsri, 0.1), 10.0)
            gmi = min(max(gmi, 0.1), 10.0)
            aqi = min(max(aqi, 0.1), 10.0)
            sgi = min(max(sgi, 0.1), 10.0)
            depi = min(max(depi, 0.1), 10.0)
            sgai = min(max(sgai, 0.1), 10.0)
            lvgi = min(max(lvgi, 0.1), 10.0)
            tata = min(max(tata, -2.0), 2.0)

            m_score = (
                -4.84
                + 0.920 * dsri
                + 0.528 * gmi
                + 0.404 * aqi
                + 0.892 * sgi
                + 0.115 * depi
                - 0.172 * sgai
                + 4.037 * tata
                + 0.0327 * lvgi
            )
            manipulation_flag = bool(m_score > -1.78)

        return {
            "status": "success",
            "altman_z_score": float(z_score) if z_score is not None else None,
            "bankruptcy_risk": bankruptcy_risk,
            "piotroski_f_score": f_score_val,
            "beneish_m_score": float(m_score) if m_score is not None else None,
            "earnings_manipulation_flag": manipulation_flag,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to compute forensic scores: {e!s}",
        }


def analyze_earnings_quality(ticker: str, periods: int) -> dict:
    """Calculates the Sloan Accrual Ratio and determines cash flow alignment.

    Args:
        ticker: The stock ticker symbol (e.g. AAPL).
        periods: The number of years of history to use (typically 5).

    Returns:
        A dictionary containing the Sloan Accrual Ratio and cash flow alignment description.
    """
    try:
        financial_data = fetch_standardized_filings(ticker, periods)
        if financial_data.get("status") == "error":
            return financial_data

        periods_list = financial_data.get("periods", [])
        if not periods_list:
            return {"status": "error", "message": "No financial data available."}

        latest = periods_list[-1]
        net_income = latest.get("net_income", 0.0)
        fcf = latest.get("free_cash_flow", 0.0)
        ta = latest.get("total_assets", 0.0)

        if ta > 0:
            sloan_ratio = (net_income - fcf) / ta
        else:
            sloan_ratio = 0.0

        alignment = "STRONG" if abs(sloan_ratio) < 0.10 else "WEAK"

        return {
            "status": "success",
            "sloan_accrual_ratio": float(sloan_ratio),
            "cash_flow_alignment": alignment,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to analyze earnings quality: {e!s}",
        }


def evaluate_graham_criteria(ticker: str, periods: int) -> dict:
    """Evaluates the asset against Benjamin Graham's strict defensive investor criteria.

    Args:
        ticker: The stock ticker symbol (e.g. AAPL).
        periods: The number of years of history to use (typically 5).

    Returns:
        A dictionary containing the results of Graham's defensive criteria evaluation.
    """
    try:
        financial_data = fetch_standardized_filings(ticker, periods)
        if financial_data.get("status") == "error":
            return financial_data

        periods_list = financial_data.get("periods", [])
        if not periods_list:
            return {"status": "error", "message": "No financial data available."}

        latest = periods_list[-1]

        # 1. Current Ratio > 2.0
        current_assets = latest.get("current_assets", 0.0)
        current_liab = latest.get("current_liabilities", 0.0)
        current_ratio = current_assets / current_liab if current_liab > 0 else 0.0
        pass_current_ratio = current_ratio > 2.0

        # 2. Long-term debt < Net current assets
        long_term_debt = latest.get("long_term_debt", 0.0)
        net_current_assets = current_assets - current_liab
        pass_debt_to_assets = long_term_debt < net_current_assets

        # 3. Positive earnings for the past years (up to 10 years or all available)
        pass_positive_earnings = True
        for p in periods_list:
            if p.get("net_income", 0.0) <= 0:
                pass_positive_earnings = False
                break

        # 4. Uninterrupted dividend payments (checks if dividends paid in each available year)
        pass_dividends = True
        for p in periods_list:
            div = p.get("dividends_paid", 0.0)
            if div >= 0:
                pass_dividends = False
                break

        graham_pass = (
            pass_current_ratio
            and pass_debt_to_assets
            and pass_positive_earnings
            and pass_dividends
        )

        return {
            "status": "success",
            "graham_defensive_pass": bool(graham_pass),
            "details": {
                "current_ratio": current_ratio,
                "pass_current_ratio": bool(pass_current_ratio),
                "long_term_debt": long_term_debt,
                "net_current_assets": net_current_assets,
                "pass_debt_to_assets": bool(pass_debt_to_assets),
                "pass_positive_earnings": bool(pass_positive_earnings),
                "pass_dividends": bool(pass_dividends),
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to evaluate Graham criteria: {e!s}",
        }
