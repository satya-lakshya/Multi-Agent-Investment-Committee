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

from app.institutional_tools import (
    analyze_block_trade_volume,
    fetch_insider_transactions,
    fetch_institutional_holdings,
)


def test_fetch_insider_transactions() -> None:
    """Verifies that fetch_insider_transactions parses insider buying activity."""
    res = fetch_insider_transactions("AAPL", days=90)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "net_buying_selling_ratio_90d" in res
    assert "key_insider_buys" in res
    assert "verdict" in res
    assert res["verdict"] in ("BULLISH", "BEARISH", "NEUTRAL")


def test_fetch_institutional_holdings() -> None:
    """Verifies that fetch_institutional_holdings parses Form 13F ownership details."""
    res = fetch_institutional_holdings("AAPL")
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "institutional_ownership_percentage" in res
    assert "quarter_over_quarter_change" in res
    assert "top_buyers" in res
    assert len(res["top_buyers"]) > 0


def test_analyze_block_trade_volume() -> None:
    """Verifies that analyze_block_trade_volume detects large volume nodes."""
    res = analyze_block_trade_volume("AAPL")
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "block_trade_anomalies" in res
    assert len(res["block_trade_anomalies"]) > 0
