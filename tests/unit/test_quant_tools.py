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

import numpy as np

from app.quant_tools import (
    calculate_factor_information_coefficient,
    test_pairs_cointegration,
    validate_signal_statistics,
)


def test_test_pairs_cointegration() -> None:
    """Verifies test_pairs_cointegration returns expected cointegration outputs."""
    # Test with simulated tickers
    res = test_pairs_cointegration("AAPL", "XLK", historical_days=30)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "tested_pair" in res
    assert "is_cointegrated" in res
    assert "spread_z_score" in res
    assert "adf_p_value" in res
    assert isinstance(res["is_cointegrated"], bool)
    assert isinstance(res["spread_z_score"], float)


def test_calculate_factor_information_coefficient() -> None:
    """Verifies Spearman Rank correlation and Information Coefficient calculations."""
    factor_data = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.05, 0.3, 0.7, 0.6]
    forward_returns = [0.02, 0.05, 0.01, 0.08, -0.01, 0.12, -0.03, 0.02, 0.06, 0.04]

    res = calculate_factor_information_coefficient(factor_data, forward_returns)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "primary_factor_ic" in res
    assert "factor_predictive_power" in res
    assert "p_value" in res

    assert isinstance(res["primary_factor_ic"], float)
    assert res["factor_predictive_power"] in ("STRONG", "WEAK", "NOISE")
    assert 0.0 <= res["p_value"] <= 1.0


def test_validate_signal_statistics() -> None:
    """Verifies Welch's t-test signal validation and confidence interval parsing."""
    np.random.seed(42)
    # Generate strategy vs benchmark returns
    strategy_returns = np.random.normal(0.0008, 0.01, 100).tolist()
    benchmark_returns = np.random.normal(0.0004, 0.009, 100).tolist()

    res = validate_signal_statistics(strategy_returns, benchmark_returns)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "t_statistic" in res
    assert "t_test_p_value" in res
    assert "statistical_significance" in res
    assert "confidence_interval" in res
    assert "noise_warning" in res

    assert isinstance(res["t_statistic"], float)
    assert isinstance(res["t_test_p_value"], float)
    assert res["statistical_significance"] in ("SIGNIFICANT", "INSIGNIFICANT")
    assert len(res["confidence_interval"]) == 2
    assert res["confidence_interval"][0] <= res["confidence_interval"][1]
