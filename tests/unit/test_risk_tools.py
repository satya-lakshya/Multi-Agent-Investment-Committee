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

from app.risk_tools import (
    analyze_debt_maturity_wall,
    calculate_tail_risk_metrics,
    extract_concentration_risks,
    run_monte_carlo_stress_test,
)


def test_calculate_tail_risk_metrics() -> None:
    """Verifies that calculate_tail_risk_metrics computes VaR, CVaR and Drawdown."""
    price_data = [100.0, 105.0, 98.0, 102.0, 95.0, 90.0, 93.0, 88.0, 85.0, 89.0]
    res = calculate_tail_risk_metrics(price_data)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "value_at_risk_95" in res
    assert "conditional_var_95" in res
    assert "max_historical_drawdown_percentage" in res

    assert isinstance(res["value_at_risk_95"], float)
    assert isinstance(res["conditional_var_95"], float)
    assert isinstance(res["max_historical_drawdown_percentage"], float)


def test_run_monte_carlo_stress_test() -> None:
    """Verifies that run_monte_carlo_stress_test computes probability of 20%+ loss."""
    res = run_monte_carlo_stress_test("AAPL", simulations=100)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "monte_carlo_probability_of_20_percent_loss" in res
    prob = res["monte_carlo_probability_of_20_percent_loss"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_analyze_debt_maturity_wall() -> None:
    """Verifies analyze_debt_maturity_wall parses balance sheet debt structure."""
    # Test case 1: structured periods data
    mock_financial_data = {
        "status": "success",
        "ticker": "AAPL",
        "periods": [
            {
                "long_term_debt": 100000000000.0,
                "current_liabilities": 50000000000.0,
                "total_assets": 350000000000.0,
            }
        ],
    }
    res = analyze_debt_maturity_wall(mock_financial_data)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "total_debt" in res
    assert "debt_to_assets_ratio" in res
    assert "debt_maturity_schedule" in res
    sched = res["debt_maturity_schedule"]
    assert "within_12_months" in sched
    assert "years_1_to_3" in sched

    # Test case 2: fallback to direct/nested keys without periods
    mock_fallback_data = {
        "long_term_debt": 9000000000.0,
        "current_liabilities": 145000000000.0,
        "total_assets": 350000000000.0,
    }
    res_fallback = analyze_debt_maturity_wall(mock_fallback_data)
    assert isinstance(res_fallback, dict)
    assert res_fallback["status"] == "success"
    assert res_fallback["total_debt"] > 0
    assert res_fallback["debt_maturity_schedule"]["within_12_months"] > 0


def test_extract_concentration_risks() -> None:
    """Verifies extract_concentration_risks detects supplier/customer concentrations."""
    text = (
        "We are exposed to high customer concentration, where a single customer "
        "comprises 30% of sales. We also have a sole supplier in Asia."
    )
    res = extract_concentration_risks(text)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "concentration_risks_found" in res
    assert len(res["concentration_risks_found"]) > 0
