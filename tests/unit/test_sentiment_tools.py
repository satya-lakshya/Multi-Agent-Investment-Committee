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

from app.sentiment_tools import (
    analyze_management_tone,
    calculate_finbert_sentiment,
    extract_catalyst_events,
    fetch_text_corpus,
)


def test_fetch_text_corpus() -> None:
    """Verifies that fetch_text_corpus retrieves news and returns transcript text."""
    res = fetch_text_corpus("AAPL", days=7)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "news" in res
    assert "earnings_transcript" in res
    assert isinstance(res["news"], list)
    assert len(res["earnings_transcript"]) > 0


def test_calculate_finbert_sentiment() -> None:
    """Verifies that calculate_finbert_sentiment calculates correct polarity score."""
    headlines = [
        "Company reports record growth and high revenues",  # strongly positive
        "Earnings call was a massive success, outperforming peers",  # strongly positive
        "Stock falls due to weak demand and contract loss",  # negative
        "Uncertain outlook reported by management",  # negative
    ]
    res = calculate_finbert_sentiment(headlines)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "finbert_polarity_score" in res
    score = res["finbert_polarity_score"]
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_analyze_management_tone() -> None:
    """Verifies analyze_management_tone counts uncertainty and litigation words."""
    transcript = (
        "We believe that global conditions could remain uncertain. "
        "Litigation risk and court disputes are possible."
    )
    res = analyze_management_tone(transcript)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert res["uncertainty_word_count"] > 0
    assert "uncertainty_word_count_change" in res
    assert res["litigation_risk_mentions"] > 0
    assert res["overall_confidence_trend"] in ["IMPROVING", "DEGRADING", "STABLE"]
    assert len(res["key_quote_flagged"]) > 0


def test_extract_catalyst_events() -> None:
    """Verifies extract_catalyst_events detects catalysts and timeframes."""
    corpus = [
        "We expect our new product launch in Q3 2026 to have a major impact.",
        "The merger approval is expected next year.",
    ]
    res = extract_catalyst_events(corpus)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert "identified_catalysts" in res
    assert len(res["identified_catalysts"]) > 0

    first = res["identified_catalysts"][0]
    assert "event_description" in first
    assert "expected_timeframe" in first
    assert first["impact_probability"] in ["LOW", "MEDIUM", "HIGH"]
