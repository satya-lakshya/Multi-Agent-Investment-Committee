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

from app.macro_tools import (
    analyze_sector_momentum,
    fetch_fred_macro_data,
    fetch_geopolitical_risk_signals,
)


def test_fetch_fred_macro_data() -> None:
    """Verifies that fetch_fred_macro_data retrieves key economic data from FRED."""
    indicators = ["DGS10", "CPIAUCSL"]
    data = fetch_fred_macro_data(indicators)

    assert isinstance(data, dict)
    for code in indicators:
        assert code in data
        assert data[code]["status"] == "success"
        assert "latest_value" in data[code]
        assert "observation_date" in data[code]
        assert data[code]["trend"] in ["RISING", "FALLING", "FLAT"]


def test_analyze_sector_momentum() -> None:
    """Verifies analyze_sector_momentum successfully calculates rolling sector returns."""
    res = analyze_sector_momentum("XLK", benchmark="SPY")

    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "sector_etf_performance_6m" in res
    assert "benchmark_performance_6m" in res
    assert res["relative_to_sp500"] in ["OUTPERFORMING", "UNDERPERFORMING", "MATCHING"]
    assert res["lifecycle_stage"] in ["GROWTH", "MATURE", "DECLINE", "ACCELERATING"]


def test_fetch_geopolitical_risk_signals() -> None:
    """Verifies fetch_geopolitical_risk_signals retrieves search news sentiment or RSS feeds."""
    res = fetch_geopolitical_risk_signals(["tariffs", "export restrictions"])

    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "articles" in res
    assert len(res["articles"]) > 0
    first_article = res["articles"][0]
    assert "title" in first_article
    assert "url" in first_article
