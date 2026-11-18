"""
GhostSignal — Tests for GDELT DOC API Collector
================================================
Includes unit tests (mocked, fast) and integration test (real API request).

Run all tests:
    python -m pytest tests/test_gdelt_collector.py -v

Run only unit tests:
    python -m pytest tests/test_gdelt_collector.py -v -m "not integration"

Run real GDELT API test:
    python -m pytest tests/test_gdelt_collector.py -v -m integration
    # or standalone:
    python tests/test_gdelt_collector.py
"""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collectors.news.gdelt_collector import (
    GDELTAPIError,
    GDELTArticle,
    GDELTCollectorError,
    GDELTDocCollector,
    GDELTQueryResult,
    GDELTRateLimitError,
    GDELTTimeoutError,
    canonicalize_url,
    format_gdelt_datetime,
    generate_record_id,
    parse_gdelt_datetime,
)


# ===========================================================================
# Unit Tests (Mocked / Fast)
# ===========================================================================

class TestCanonicalizeUrl(unittest.TestCase):
    """Test URL normalization and canonicalization logic."""

    def test_basic_normalization(self):
        url = "HTTP://Example.COM/Path/To/Page/?a=1"
        self.assertEqual(
            canonicalize_url(url),
            "http://example.com/Path/To/Page?a=1"
        )

    def test_strip_tracking_parameters(self):
        url = (
            "https://news.example.com/article?"
            "utm_source=twitter&utm_medium=social&utm_campaign=winter"
            "&fbclid=IwAR123&article_id=456&ocid=789"
        )
        expected = "https://news.example.com/article?article_id=456"
        self.assertEqual(canonicalize_url(url), expected)

    def test_sort_query_parameters(self):
        url1 = "https://example.com/page?z=1&a=2"
        url2 = "https://example.com/page?a=2&z=1"
        self.assertEqual(canonicalize_url(url1), canonicalize_url(url2))

    def test_strip_fragments(self):
        url = "https://example.com/page#section-1"
        self.assertEqual(canonicalize_url(url), "https://example.com/page")

    def test_handle_empty_and_invalid(self):
        self.assertEqual(canonicalize_url(""), "")
        self.assertEqual(canonicalize_url(None), "")


class TestDateParsing(unittest.TestCase):
    """Test GDELT date format handling."""

    def test_parse_iso_z_format(self):
        parsed = parse_gdelt_datetime("20261025T033000Z")
        self.assertEqual(parsed, "2026-10-25T03:30:00+00:00")

    def test_parse_14_digit_format(self):
        parsed = parse_gdelt_datetime("20261025033000")
        self.assertEqual(parsed, "2026-10-25T03:30:00+00:00")

    def test_parse_8_digit_date(self):
        parsed = parse_gdelt_datetime("20261025")
        self.assertEqual(parsed, "2026-10-25T00:00:00+00:00")

    def test_parse_invalid(self):
        self.assertIsNone(parse_gdelt_datetime("not-a-date"))
        self.assertIsNone(parse_gdelt_datetime(""))
        self.assertIsNone(parse_gdelt_datetime(None))

    def test_format_gdelt_datetime(self):
        dt = datetime(2026, 11, 18, 9, 30, 0)
        self.assertEqual(format_gdelt_datetime(dt), "20261118093000")
        self.assertEqual(format_gdelt_datetime("2026-11-18"), "20261118000000")


class TestGDELTDocCollectorUnit(unittest.TestCase):
    """Unit tests for GDELTDocCollector query building, deduplication and mock responses."""

    def setUp(self):
        self.collector = GDELTDocCollector(min_request_interval=0.0, retry_backoff=0.01)

    def test_build_params_clamping(self):
        params = self.collector.build_params("cyberattack", max_records=999)
        self.assertEqual(params["maxrecords"], 250)
        self.assertEqual(params["query"], "cyberattack")
        self.assertEqual(params["mode"], "artlist")
        self.assertEqual(params["format"], "json")

    def test_build_params_with_filters(self):
        params = self.collector.build_params(
            "deepfake",
            domain="reuters.com",
            language="eng",
            country="US",
            timespan="24h"
        )
        self.assertIn("domain:reuters.com", params["query"])
        self.assertIn("sourcelang:eng", params["query"])
        self.assertIn("sourcecountry:US", params["query"])
        self.assertEqual(params["timespan"], "24h")

    def test_deduplication_via_canonical_urls(self):
        # Sample response containing identical articles under different URLs with tracking tags
        mock_response_data = {
            "articles": [
                {
                    "url": "https://example.com/news/123?utm_source=twitter",
                    "title": "Disinformation in Election",
                    "seendate": "20261115T120000Z",
                    "domain": "example.com",
                    "language": "English",
                    "sourcecountry": "United States",
                },
                {
                    "url": "https://example.com/news/123?utm_source=facebook",
                    "title": "Disinformation in Election",
                    "seendate": "20261115T120000Z",
                    "domain": "example.com",
                    "language": "English",
                    "sourcecountry": "United States",
                },
                {
                    "url": "https://example.com/news/456",
                    "title": "Unrelated Story",
                    "seendate": "20261116T080000Z",
                    "domain": "example.com",
                    "language": "English",
                    "sourcecountry": "United States",
                },
            ]
        }

        with patch.object(self.collector, "_execute_request", return_value=mock_response_data):
            result = self.collector.collect("election", deduplicate=True)

        self.assertEqual(result.total_records, 2)
        self.assertEqual(len(result.articles), 2)
        urls = [art.url for art in result.articles]
        self.assertIn("https://example.com/news/123?utm_source=twitter", urls)
        self.assertIn("https://example.com/news/456", urls)

    def test_empty_results_handling(self):
        with patch.object(self.collector, "_execute_request", return_value={"articles": []}):
            result = self.collector.collect("nonexistenttermXYZ123456")

        self.assertEqual(result.total_records, 0)
        self.assertEqual(len(result.articles), 0)
        self.assertEqual(len(result.errors), 0)

    def test_no_fabrication_contract(self):
        mock_response_data = {
            "articles": [
                {
                    "url": "https://bbc.com/news/test-1",
                    "title": "Sample Verified Headline",
                    "seendate": "20261117T100000Z",
                    "domain": "bbc.com",
                    "language": "English",
                    "sourcecountry": "United Kingdom",
                }
            ]
        }
        with patch.object(self.collector, "_execute_request", return_value=mock_response_data):
            result = self.collector.collect("test")

        art = result.articles[0]
        # Main content is exactly the title, no synthetic bodies
        self.assertEqual(art.content, "Sample Verified Headline")
        self.assertEqual(art.title, "Sample Verified Headline")
        # Optional missing attributes remain None per data contract
        self.assertIsNone(art.author)
        self.assertIsNone(art.product)
        self.assertIsNone(art.rating)
        self.assertIsNone(art.likes)
        self.assertIsNone(art.comments)
        self.assertIsNone(art.shares)
        self.assertIsNone(art.views)
        self.assertIsNone(art.label)
        # Provenance is preserved
        self.assertEqual(art.metadata["provenance"]["api_endpoint"], GDELTDocCollector.ENDPOINT)
        self.assertEqual(art.metadata["provenance"]["source_domain"], "bbc.com")

    def test_timeout_and_error_handling(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")
        collector = GDELTDocCollector(
            session=mock_session,
            max_retries=1,
            retry_backoff=0.01,
            min_request_interval=0.0,
        )

        # raise_on_error=False (default): graceful container return with error logged
        res = collector.collect("timeout_test", raise_on_error=False)
        self.assertEqual(res.total_records, 0)
        self.assertTrue(len(res.errors) > 0)
        self.assertIn("timed out", res.errors[0])

        # raise_on_error=True: should raise GDELTTimeoutError
        with self.assertRaises(GDELTTimeoutError):
            collector.collect("timeout_test", raise_on_error=True)


# ===========================================================================
# Integration Test (Real API Request)
# ===========================================================================

@pytest.mark.integration
def test_real_gdelt_api():
    """
    Integration test making a real query to GDELT DOC 2.0 API.
    Verifies that the live endpoint is reachable, accepts queries,
    and returns compliant structured JSON.
    """
    collector = GDELTDocCollector(min_request_interval=5.5, timeout=20.0)
    query = "cybersecurity"

    result = collector.collect(
        query=query,
        max_records=2,
        timespan="7d",
        raise_on_error=False,
    )

    print(f"\n[Real API Test] Query: '{query}'")
    print(f"[Real API Test] Articles retrieved: {result.total_records}")

    if result.errors:
        print(f"[Real API Test] Note: API returned notice/error: {result.errors}")

    # Structured JSON output verification
    json_output = result.to_json(indent=2)
    assert isinstance(json_output, str)
    assert len(json_output) > 0

    if result.total_records > 0:
        art = result.articles[0]
        assert art.source == "gdelt"
        assert art.id.startswith("gdelt_")
        assert art.url is not None
        assert "provenance" in art.metadata
        assert art.metadata["provenance"]["collector"] == "GDELTDocCollector"
        print(f"[Real API Test] Sample Article Title: {art.title}")
        print(f"[Real API Test] Sample Domain: {art.metadata.get('domain')}")


if __name__ == "__main__":
    print("Running GDELT collector unit tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCanonicalizeUrl)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDateParsing))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGDELTDocCollectorUnit))
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)

    if not res.wasSuccessful():
        sys.exit(1)

    print("\nRunning GDELT real API live test...")
    test_real_gdelt_api()
