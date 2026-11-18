"""
GhostSignal — Tests for Investigation Service & Automated Workflow
===================================================================
Verifies:
- Automated concurrent execution of both GDELT and Hacker News collectors.
- Execution without independent CLI commands.
- Sentiment analysis applied to all collected records with predictions & confidence attached.
- Calculation of positive and negative counts separately for news and discussions.
- Partial failure handling (GDELT fails or HN fails).
- Total failure handling (both collectors fail).
- Retention of source identity ('gdelt' vs 'hackernews') and provenance.
- Integration with Flask /api/investigate endpoint.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app
from src.collectors.discussions.hackernews_collector import (
    HackerNewsAPIError,
    HackerNewsQueryResult,
    HackerNewsRecord,
)
from src.collectors.news.gdelt_collector import (
    GDELTAPIError,
    GDELTArticle,
    GDELTQueryResult,
)
from src.services.investigation_service import (
    InvestigationResponse,
    InvestigationService,
    attach_sentiment_to_records,
    build_timeline_from_records,
    compute_sentiment_stats,
)


def _make_sample_gdelt_record(record_id="gdelt_1001", title="Cybersecurity Alert: Great improvements made"):
    return GDELTArticle(
        id=record_id,
        source="gdelt",
        content=title,
        title=title,
        url="https://news.example.com/cyber",
        published_at="2026-11-18T01:00:00+00:00",
        retrieved_at="2026-11-18T04:00:00+00:00",
        author=None,
        metadata={
            "domain": "example.com",
            "provenance": {
                "collector": "GDELTDocCollector",
                "api_endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
                "source_domain": "example.com",
            },
        },
    )


def _make_sample_hn_record(record_id="hn_2001", title="Critical zero day flaw discovered", item_type="story"):
    return HackerNewsRecord(
        id=record_id,
        source="hackernews",
        content=title,
        title=title,
        url="https://news.ycombinator.com/item?id=2001",
        published_at="2026-11-18T02:00:00+00:00",
        retrieved_at="2026-11-18T04:00:00+00:00",
        author="sec_expert",
        likes=45,
        comments=18,
        metadata={
            "item_type": item_type,
            "object_id": record_id.replace("hn_", ""),
            "provenance": {
                "collector": "HackerNewsCollector",
                "api_endpoint": "https://hn.algolia.com/api/v1/search_by_date",
                "tag": item_type,
            },
        },
    )


def _make_mock_classifier(pred_map=None):
    """Create a mock classifier returning configurable sentiment per text."""
    clf = MagicMock()
    default_pred_map = pred_map or {}

    def _predict(text):
        res = MagicMock()
        label = default_pred_map.get(text, "Positive" if "great" in text.lower() else "Negative")
        proba = {"Positive": 0.85, "Negative": 0.15} if label == "Positive" else {"Positive": 0.20, "Negative": 0.80}
        res.sentiment.label = label
        res.sentiment.probabilities = proba
        res.sentiment.is_uncertain = False
        res.sentiment.model_available = True
        res.sentiment.caveat = "Mock caveat"
        res.authenticity.label = "HONEST"
        res.authenticity.probabilities = {"HONEST": 0.90, "FAKE": 0.10}
        res.authenticity.is_uncertain = False
        return res

    clf.predict.side_effect = _predict
    return clf


class TestInvestigationService(unittest.TestCase):
    """Test suite for automated multi-source investigation service."""

    def setUp(self):
        self.mock_gdelt = MagicMock()
        self.mock_hn = MagicMock()
        self.service = InvestigationService(
            gdelt_collector=self.mock_gdelt,
            hn_collector=self.mock_hn,
            max_workers=2,
        )

    def test_investigation_invokes_both_collectors_automatically(self):
        """Starting an investigation must invoke both GDELT and Hacker News without CLI intervention."""
        gdelt_art = _make_sample_gdelt_record()
        hn_story = _make_sample_hn_record(item_type="story")
        hn_comm = _make_sample_hn_record(record_id="hn_2002", title="Useful comment", item_type="comment")

        self.mock_gdelt.collect.return_value = GDELTQueryResult(
            query="disinformation",
            total_records=1,
            retrieved_at="2026-11-18T04:00:00+00:00",
            articles=[gdelt_art],
        )
        self.mock_hn.collect.return_value = HackerNewsQueryResult(
            query="disinformation",
            total_records=2,
            stories_count=1,
            comments_count=1,
            retrieved_at="2026-11-18T04:00:00+00:00",
            records=[hn_story, hn_comm],
        )

        res = self.service.investigate("disinformation")

        # Verify both collectors were called
        self.mock_gdelt.collect.assert_called_once()
        self.mock_hn.collect.assert_called_once()
        self.assertEqual(self.mock_gdelt.collect.call_args.kwargs["query"], "disinformation")
        self.assertEqual(self.mock_hn.collect.call_args.kwargs["query"], "disinformation")

        # Verify output
        self.assertEqual(res.status, "success")
        self.assertEqual(res.total_records, 3)
        self.assertEqual(res.news_count, 1)
        self.assertEqual(res.discussions_count, 2)
        self.assertEqual(len(res.records), 3)
        self.assertEqual(len(res.collector_errors), 0)

    def test_sentiment_attached_to_records_and_stats_computed(self):
        """Pass collected text into sentiment analyser and attach prediction + confidence to records."""
        gdelt_art = _make_sample_gdelt_record(title="Great breakthrough in technology")
        hn_story = _make_sample_hn_record(title="Critical security vulnerability flaw")

        self.mock_gdelt.collect.return_value = GDELTQueryResult(
            query="tech", total_records=1, retrieved_at="2026-11-18T04:00:00+00:00", articles=[gdelt_art]
        )
        self.mock_hn.collect.return_value = HackerNewsQueryResult(
            query="tech", total_records=1, stories_count=1, comments_count=0, retrieved_at="2026-11-18T04:00:00+00:00", records=[hn_story]
        )

        mock_clf = _make_mock_classifier()
        res = self.service.investigate("tech", classifier=mock_clf)

        # Check predictions attached to records
        news_item = res.news_records[0]
        self.assertEqual(news_item["predicted_label"], "Positive")
        self.assertEqual(news_item["confidence"], 0.85)

        disc_item = res.discussion_records[0]
        self.assertEqual(disc_item["predicted_label"], "Negative")
        self.assertEqual(disc_item["confidence"], 0.80)

        # Check separate news and discussions sentiment statistics
        self.assertEqual(res.sentiment_stats["news"]["positive"], 1)
        self.assertEqual(res.sentiment_stats["news"]["negative"], 0)
        self.assertEqual(res.sentiment_stats["discussions"]["positive"], 0)
        self.assertEqual(res.sentiment_stats["discussions"]["negative"], 1)
        self.assertEqual(res.sentiment_stats["totals"]["positive"], 1)
        self.assertEqual(res.sentiment_stats["totals"]["negative"], 1)

    def test_results_retain_source_identity_and_provenance(self):
        """Combined records must retain their respective source names and provenance."""
        gdelt_art = _make_sample_gdelt_record()
        hn_rec = _make_sample_hn_record()

        self.mock_gdelt.collect.return_value = GDELTQueryResult(
            query="test", total_records=1, retrieved_at="2026-11-18T04:00:00+00:00", articles=[gdelt_art]
        )
        self.mock_hn.collect.return_value = HackerNewsQueryResult(
            query="test", total_records=1, stories_count=1, comments_count=0, retrieved_at="2026-11-18T04:00:00+00:00", records=[hn_rec]
        )

        res = self.service.investigate("test")

        sources = [r["source"] for r in res.records]
        self.assertIn("gdelt", sources)
        self.assertIn("hackernews", sources)

        # Check provenance
        gdelt_item = next(r for r in res.records if r["source"] == "gdelt")
        hn_item = next(r for r in res.records if r["source"] == "hackernews")
        self.assertEqual(gdelt_item["metadata"]["provenance"]["collector"], "GDELTDocCollector")
        self.assertEqual(hn_item["metadata"]["provenance"]["collector"], "HackerNewsCollector")

    def test_partial_failure_gdelt_fails_hn_succeeds(self):
        """When GDELT encounters an error, Hacker News results are still returned with status='partial'."""
        self.mock_gdelt.collect.side_effect = GDELTAPIError("GDELT HTTP 503 Service Unavailable")
        hn_rec = _make_sample_hn_record()
        self.mock_hn.collect.return_value = HackerNewsQueryResult(
            query="test", total_records=1, stories_count=1, comments_count=0, retrieved_at="2026-11-18T04:00:00+00:00", records=[hn_rec]
        )

        res = self.service.investigate("test")

        self.assertEqual(res.status, "partial")
        self.assertEqual(res.news_count, 0)
        self.assertEqual(res.discussions_count, 1)
        self.assertEqual(res.total_records, 1)
        self.assertIn("gdelt", res.collector_errors)
        self.assertIn("503", res.collector_errors["gdelt"])
        self.assertNotIn("hackernews", res.collector_errors)

    def test_partial_failure_hn_fails_gdelt_succeeds(self):
        """When Hacker News encounters an error, GDELT results are still returned with status='partial'."""
        gdelt_art = _make_sample_gdelt_record()
        self.mock_gdelt.collect.return_value = GDELTQueryResult(
            query="test", total_records=1, retrieved_at="2026-11-18T04:00:00+00:00", articles=[gdelt_art]
        )
        self.mock_hn.collect.side_effect = HackerNewsAPIError("Algolia HTTP 429 Rate Limit")

        res = self.service.investigate("test")

        self.assertEqual(res.status, "partial")
        self.assertEqual(res.news_count, 1)
        self.assertEqual(res.discussions_count, 0)
        self.assertEqual(res.total_records, 1)
        self.assertIn("hackernews", res.collector_errors)
        self.assertNotIn("gdelt", res.collector_errors)

    def test_total_failure_returns_error_status(self):
        """When both collectors fail, returns status='error' with diagnostic error details."""
        self.mock_gdelt.collect.side_effect = GDELTAPIError("GDELT connection failed")
        self.mock_hn.collect.side_effect = HackerNewsAPIError("HN timeout")

        res = self.service.investigate("test")

        self.assertEqual(res.status, "error")
        self.assertEqual(res.total_records, 0)
        self.assertIn("gdelt", res.collector_errors)
        self.assertIn("hackernews", res.collector_errors)

    def test_build_timeline_from_records_with_sentiment(self):
        """Activity timeline reflects real publication dates and sentiment labels from records."""
        records = [
            {"published_at": "2026-11-18T10:00:00+00:00", "predicted_label": "Positive"},
            {"published_at": "2026-11-18T12:00:00+00:00", "predicted_label": "Negative"},
            {"published_at": "2026-11-17T08:00:00+00:00", "predicted_label": "Positive"},
        ]
        timeline = build_timeline_from_records(records, days=7, pos_prob=0.5)
        self.assertEqual(len(timeline), 7)
        total_in_timeline = sum(d["all"] for d in timeline)
        self.assertEqual(total_in_timeline, 3)


class TestAppInvestigationEndpoint(unittest.TestCase):
    """Test Flask /api/investigate integration with InvestigationService."""

    def setUp(self):
        self.client = app.test_client()

    def test_endpoint_success_flow(self):
        gdelt_rec = _make_sample_gdelt_record().model_dump()
        gdelt_rec["predicted_label"] = "Positive"
        gdelt_rec["confidence"] = 0.85

        hn_rec = _make_sample_hn_record().model_dump()
        hn_rec["predicted_label"] = "Negative"
        hn_rec["confidence"] = 0.80

        mock_response = InvestigationResponse(
            investigation_id="GS-TEST",
            query="deepseek",
            status="success",
            total_records=2,
            news_count=1,
            discussions_count=1,
            news_records=[gdelt_rec],
            discussion_records=[hn_rec],
            records=[gdelt_rec, hn_rec],
            sentiment_stats={
                "news": {"total": 1, "positive": 1, "negative": 0, "uncertain": 0, "positive_pct": 100},
                "discussions": {"total": 1, "positive": 0, "negative": 1, "uncertain": 0, "positive_pct": 0},
                "totals": {"total": 2, "positive": 1, "negative": 1, "uncertain": 0, "positive_pct": 50},
            },
            collector_errors={},
            retrieved_at="2026-11-18T04:00:00+00:00",
        )

        with patch("app.get_investigation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.investigate.return_value = mock_response
            mock_get_service.return_value = mock_service

            resp = self.client.post("/api/investigate", json={"topic": "deepseek"})

        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["investigation_id"], "GS-TEST")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["news_count"], 1)
        self.assertEqual(data["discussions_count"], 1)
        self.assertEqual(data["metrics"]["records_total"], 2)
        self.assertEqual(data["metrics"]["news_positive"], 1)
        self.assertEqual(data["metrics"]["discussions_negative"], 1)
        self.assertIn("sentiment_stats", data)
        self.assertEqual(len(data["records"]), 2)
        self.assertEqual(data["records"][0]["predicted_label"], "Positive")
        self.assertEqual(data["records"][1]["predicted_label"], "Negative")
        self.assertIn("timeline", data)
        self.assertIn("sentiment", data)
        self.assertIn("authenticity", data)

    def test_endpoint_total_failure_flow(self):
        mock_response = InvestigationResponse(
            investigation_id="GS-FAIL",
            query="failed_query",
            status="error",
            total_records=0,
            news_count=0,
            discussions_count=0,
            news_records=[],
            discussion_records=[],
            records=[],
            sentiment_stats={},
            collector_errors={"gdelt": "Rate limit 429", "hackernews": "Connection timeout"},
            retrieved_at="2026-11-18T04:00:00+00:00",
        )

        with patch("app.get_investigation_service") as mock_get_service:
            mock_service = MagicMock()
            mock_service.investigate.return_value = mock_response
            mock_get_service.return_value = mock_service

            resp = self.client.post("/api/investigate", json={"topic": "failed_query"})

        self.assertEqual(resp.status_code, 502)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")
        self.assertIn("All collection sources failed", data["error"])
        self.assertEqual(data["metrics"]["records_total"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
