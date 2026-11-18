"""
GhostSignal — Tests for Bluesky Discussion Collector
======================================================
Offline unit tests covering:
- Parameter building
- Post normalisation (text, author, timestamps, engagement, external links)
- Missing fields: no fabrication
- Deduplication by record ID
- Empty results handling
- HTTP 403 (BlueskyForbiddenError)
- HTTP 429 retry exhaustion (BlueskyRateLimitError)
- Generic HTTP error
- Network timeout
- Graceful error capture (raise_on_error=False)
- Schema compatibility with GDELTArticle and HackerNewsRecord

Optional live integration test (skipped by default):
    $env:RUN_BSKY_LIVE_TEST="1"
    python -m pytest tests/test_bluesky_collector.py -v -k "test_live_bluesky_api"

Run offline tests:
    python -m pytest tests/test_bluesky_collector.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collectors.discussions.bluesky_collector import (
    BlueskyAPIError,
    BlueskyCollector,
    BlueskyCollectorError,
    BlueskyForbiddenError,
    BlueskyQueryResult,
    BlueskyRateLimitError,
    BlueskyRecord,
    BlueskyTimeoutError,
    build_bsky_permalink,
    extract_external_links,
    parse_bsky_rkey,
)
from src.collectors.discussions.hackernews_collector import HackerNewsRecord
from src.collectors.news.gdelt_collector import GDELTArticle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_post(
    uri: str = "at://did:plc:abc123/app.bsky.feed.post/3n2testpost01",
    cid: str = "bafytest001",
    text: str = "GhostSignal is a great OSINT tool for tracking narratives.",
    handle: str = "alice.bsky.social",
    did: str = "did:plc:abc123",
    display_name: str = "Alice Researcher",
    created_at: str = "2026-11-18T10:00:00.000Z",
    like_count: int = 42,
    reply_count: int = 7,
    repost_count: int = 5,
    embed: dict | None = None,
    facets: list | None = None,
) -> dict:
    """Build a minimal realistic Bluesky post dict as returned by the XRPC API."""
    record: dict = {
        "$type": "app.bsky.feed.post",
        "text": text,
        "createdAt": created_at,
    }
    if embed:
        record["embed"] = embed
    if facets:
        record["facets"] = facets

    return {
        "uri": uri,
        "cid": cid,
        "author": {
            "did": did,
            "handle": handle,
            "displayName": display_name,
        },
        "record": record,
        "likeCount": like_count,
        "replyCount": reply_count,
        "repostCount": repost_count,
        "indexedAt": created_at,
    }


def _api_response(posts: list, hits_total: int | None = None) -> dict:
    """Wrap posts in a minimal Bluesky API response envelope."""
    result: dict = {"posts": posts}
    if hits_total is not None:
        result["hitsTotal"] = hits_total
    return result


def _mock_http_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    """Return a mock requests.Response with controlled status_code and body."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text if text else ("{}" if json_body is None else "")
    if json_body is not None:
        mock_resp.json.return_value = json_body
    else:
        mock_resp.json.side_effect = ValueError("No JSON")
    return mock_resp


# ===========================================================================
# 1. Utility Function Tests
# ===========================================================================
class TestBlueskyUtilityFunctions(unittest.TestCase):

    def test_parse_bsky_rkey_standard_uri(self):
        uri = "at://did:plc:wcgbb35w2zjdq4gg42qjqqn3/app.bsky.feed.post/3n2kuor6tsi26"
        self.assertEqual(parse_bsky_rkey(uri), "3n2kuor6tsi26")

    def test_parse_bsky_rkey_trailing_slash(self):
        uri = "at://did:plc:abc/app.bsky.feed.post/rkey99/"
        self.assertEqual(parse_bsky_rkey(uri), "rkey99")

    def test_parse_bsky_rkey_none_and_empty(self):
        self.assertEqual(parse_bsky_rkey(None), "")
        self.assertEqual(parse_bsky_rkey(""), "")

    def test_build_bsky_permalink(self):
        url = build_bsky_permalink("alice.bsky.social", "3n2kuor6tsi26")
        self.assertEqual(url, "https://bsky.app/profile/alice.bsky.social/post/3n2kuor6tsi26")

    def test_build_bsky_permalink_did_fallback(self):
        url = build_bsky_permalink("did:plc:abc123", "rkey1")
        self.assertIn("did:plc:abc123", url)
        self.assertIn("rkey1", url)

    def test_extract_external_links_embed_card(self):
        post_data = {
            "record": {
                "embed": {
                    "$type": "app.bsky.embed.external",
                    "external": {"uri": "https://example.com/article"},
                },
                "facets": [],
            }
        }
        links = extract_external_links(post_data)
        self.assertEqual(links, ["https://example.com/article"])

    def test_extract_external_links_rich_text_facets(self):
        post_data = {
            "record": {
                "embed": {},
                "facets": [
                    {
                        "features": [
                            {
                                "$type": "app.bsky.richtext.facet#link",
                                "uri": "https://news.example.com/story",
                            }
                        ]
                    }
                ],
            }
        }
        links = extract_external_links(post_data)
        self.assertIn("https://news.example.com/story", links)

    def test_extract_external_links_deduplication(self):
        """Same URL appearing in both embed and facets should appear once."""
        post_data = {
            "record": {
                "embed": {
                    "$type": "app.bsky.embed.external",
                    "external": {"uri": "https://dup.example.com"},
                },
                "facets": [
                    {
                        "features": [
                            {"$type": "app.bsky.richtext.facet#link", "uri": "https://dup.example.com"}
                        ]
                    }
                ],
            }
        }
        links = extract_external_links(post_data)
        self.assertEqual(links.count("https://dup.example.com"), 1)

    def test_extract_external_links_empty(self):
        self.assertEqual(extract_external_links({}), [])
        self.assertEqual(extract_external_links({"record": {}}), [])


# ===========================================================================
# 2. Collector Parameter Building
# ===========================================================================
class TestBlueskyCollectorParams(unittest.TestCase):

    def setUp(self):
        self.collector = BlueskyCollector(timeout=5.0, retry_backoff=0.01)

    def test_build_params_defaults(self):
        params = self.collector.build_params("ghostsignal")
        self.assertEqual(params["q"], "ghostsignal")
        self.assertEqual(params["limit"], 50)
        self.assertEqual(params["sort"], "latest")
        self.assertNotIn("since", params)
        self.assertNotIn("until", params)

    def test_build_params_clamped_limit(self):
        params = self.collector.build_params("test", limit=200)
        self.assertEqual(params["limit"], 100)  # max 100

    def test_build_params_min_limit(self):
        params = self.collector.build_params("test", limit=0)
        self.assertEqual(params["limit"], 1)  # min 1

    def test_build_params_with_dates(self):
        params = self.collector.build_params(
            "misinformation", limit=25, sort="top",
            since="2026-11-01T00:00:00Z", until="2026-11-18T00:00:00Z",
        )
        self.assertEqual(params["sort"], "top")
        self.assertEqual(params["since"], "2026-11-01T00:00:00Z")
        self.assertEqual(params["until"], "2026-11-18T00:00:00Z")

    def test_build_params_empty_query_raises(self):
        with self.assertRaises(ValueError):
            self.collector.build_params("   ")


# ===========================================================================
# 3. Post Normalisation
# ===========================================================================
class TestBlueskyPostNormalisation(unittest.TestCase):

    def setUp(self):
        self.collector = BlueskyCollector(timeout=5.0, retry_backoff=0.01)
        self.retrieved_at = "2026-11-18T12:00:00+00:00"

    def test_normalise_full_post(self):
        post = _make_post()
        rec = self.collector._normalize_post(post, "test query", self.retrieved_at)

        self.assertIsNotNone(rec)
        self.assertTrue(rec.id.startswith("bsky_"))
        self.assertEqual(rec.source, "bluesky")
        self.assertIn("GhostSignal", rec.content)
        self.assertEqual(rec.author, "alice.bsky.social")
        self.assertEqual(rec.likes, 42)
        self.assertEqual(rec.comments, 7)
        self.assertEqual(rec.shares, 5)
        self.assertIsNone(rec.views)
        self.assertIsNone(rec.title)
        self.assertIsNone(rec.product)
        self.assertIsNone(rec.rating)
        self.assertEqual(rec.retrieved_at, self.retrieved_at)
        self.assertEqual(rec.metadata["handle"], "alice.bsky.social")
        self.assertEqual(rec.metadata["did"], "did:plc:abc123")
        self.assertEqual(rec.metadata["display_name"], "Alice Researcher")
        self.assertEqual(rec.metadata["provenance"]["collector"], "BlueskyCollector")
        self.assertEqual(rec.metadata["provenance"]["query"], "test query")

    def test_normalise_post_permalink_format(self):
        post = _make_post(
            uri="at://did:plc:abc123/app.bsky.feed.post/3n2testpost01",
            handle="alice.bsky.social",
        )
        rec = self.collector._normalize_post(post, "test", self.retrieved_at)
        self.assertEqual(rec.url, "https://bsky.app/profile/alice.bsky.social/post/3n2testpost01")

    def test_normalise_post_published_at_from_created_at(self):
        post = _make_post(created_at="2026-11-17T08:30:00.000Z")
        rec = self.collector._normalize_post(post, "test", self.retrieved_at)
        # Should preserve the ISO string from createdAt
        self.assertIsNotNone(rec.published_at)
        self.assertIn("2026-11-17", rec.published_at)

    def test_normalise_post_external_link_captured(self):
        post = _make_post(
            embed={
                "$type": "app.bsky.embed.external",
                "external": {"uri": "https://reuters.com/story"},
            }
        )
        rec = self.collector._normalize_post(post, "test", self.retrieved_at)
        self.assertIn("https://reuters.com/story", rec.metadata["external_links"])

    def test_normalise_post_missing_fields_no_fabrication(self):
        """Sparse post: all optional fields must remain None, not fabricated."""
        sparse_post = {
            "uri": "at://did:plc:xyz/app.bsky.feed.post/sparsekey",
            "cid": "bafysparse",
            "author": {},
            "record": {"$type": "app.bsky.feed.post"},
            "indexedAt": None,
        }
        rec = self.collector._normalize_post(sparse_post, "test", self.retrieved_at)

        self.assertIsNotNone(rec)
        self.assertEqual(rec.content, "")
        self.assertIsNone(rec.author)
        self.assertIsNone(rec.likes)
        self.assertIsNone(rec.comments)
        self.assertIsNone(rec.shares)
        self.assertIsNone(rec.published_at)
        self.assertIsNone(rec.product)
        self.assertIsNone(rec.rating)
        self.assertIsNone(rec.label)
        self.assertIsNone(rec.predicted_label)
        self.assertIsNone(rec.confidence)

    def test_normalise_post_non_dict_returns_none(self):
        self.assertIsNone(self.collector._normalize_post("not a dict", "test", self.retrieved_at))
        self.assertIsNone(self.collector._normalize_post(None, "test", self.retrieved_at))

    def test_record_id_uses_rkey(self):
        post = _make_post(uri="at://did:plc:abc/app.bsky.feed.post/myrkey123")
        rec = self.collector._normalize_post(post, "test", self.retrieved_at)
        self.assertEqual(rec.id, "bsky_myrkey123")

    def test_record_id_falls_back_to_cid(self):
        """When URI has no parseable rkey, fall back to CID prefix."""
        post = _make_post(uri=None, cid="bafycid001abcdef")
        rec = self.collector._normalize_post(post, "test", self.retrieved_at)
        self.assertTrue(rec.id.startswith("bsky_"))
        self.assertGreater(len(rec.id), len("bsky_"))


# ===========================================================================
# 4. Collection Pipeline: Results, Deduplication, Error Handling
# ===========================================================================
class TestBlueskyCollectorPipeline(unittest.TestCase):

    def setUp(self):
        self.collector = BlueskyCollector(timeout=5.0, retry_backoff=0.01)

    def test_collect_returns_structured_result(self):
        post1 = _make_post(uri="at://did:plc:a/app.bsky.feed.post/key001", text="First post")
        post2 = _make_post(uri="at://did:plc:b/app.bsky.feed.post/key002", text="Second post")
        payload = _api_response([post1, post2])

        with patch.object(self.collector, "_execute_request", return_value=payload):
            result = self.collector.collect("narrative intelligence")

        self.assertIsInstance(result, BlueskyQueryResult)
        self.assertEqual(result.total_records, 2)
        self.assertEqual(result.query, "narrative intelligence")
        self.assertEqual(len(result.errors), 0)
        ids = [r.id for r in result.records]
        self.assertIn("bsky_key001", ids)
        self.assertIn("bsky_key002", ids)

    def test_collect_deduplication_by_id(self):
        """Two posts with the same URI produce only one record."""
        post = _make_post(uri="at://did:plc:a/app.bsky.feed.post/dupkey")
        payload = _api_response([post, post])  # exact duplicate

        with patch.object(self.collector, "_execute_request", return_value=payload):
            result = self.collector.collect("dup test")

        self.assertEqual(result.total_records, 1)

    def test_collect_empty_results(self):
        """Zero posts returns a valid empty result with no errors."""
        with patch.object(self.collector, "_execute_request", return_value={"posts": []}):
            result = self.collector.collect("obscure_topic_xyz_99999")

        self.assertEqual(result.total_records, 0)
        self.assertEqual(len(result.records), 0)
        self.assertEqual(len(result.errors), 0)

    def test_collect_empty_body_response(self):
        """API returning empty body is handled gracefully."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        collector = BlueskyCollector(session=mock_session, max_retries=0)

        result = collector.collect("empty body test", raise_on_error=False)
        self.assertEqual(result.total_records, 0)

    def test_collect_to_records_dicts(self):
        """to_records() returns plain dicts with the expected schema fields."""
        post = _make_post()
        with patch.object(self.collector, "_execute_request", return_value=_api_response([post])):
            result = self.collector.collect("test")

        records = result.to_records()
        self.assertEqual(len(records), 1)
        rec = records[0]
        for field in ("id", "source", "content", "url", "published_at", "retrieved_at",
                      "author", "likes", "comments", "shares", "metadata"):
            self.assertIn(field, rec, f"Missing field '{field}' in to_records() output")

    # -----------------------------------------------------------------------
    # HTTP error scenarios
    # -----------------------------------------------------------------------

    def test_http_403_raises_forbidden(self):
        """HTTP 403 must raise BlueskyForbiddenError immediately (no retry)."""
        mock_resp = _mock_http_response(403, text="Forbidden by Cloudflare")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        collector = BlueskyCollector(session=mock_session, max_retries=2, retry_backoff=0.01)

        with self.assertRaises(BlueskyForbiddenError):
            collector._execute_request({"q": "test", "limit": 10, "sort": "latest"})

        # 403 must not be retried — only one HTTP call made
        self.assertEqual(mock_session.get.call_count, 1)

    def test_http_403_captured_gracefully(self):
        """HTTP 403 is captured in errors list when raise_on_error=False."""
        mock_resp = _mock_http_response(403, text="Forbidden")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        collector = BlueskyCollector(session=mock_session, max_retries=0, retry_backoff=0.01)

        result = collector.collect("forbidden test", raise_on_error=False)
        self.assertEqual(result.total_records, 0)
        self.assertTrue(len(result.errors) > 0)
        self.assertTrue(any("403" in e or "forbidden" in e.lower() for e in result.errors))

    def test_http_429_raises_rate_limit_after_retries(self):
        """HTTP 429 exhausts retries and raises BlueskyRateLimitError."""
        mock_resp = _mock_http_response(429, text="Rate limited")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        collector = BlueskyCollector(session=mock_session, max_retries=1, retry_backoff=0.01)

        with self.assertRaises(BlueskyRateLimitError):
            collector._execute_request({"q": "test", "limit": 10, "sort": "latest"})

    def test_http_429_captured_gracefully(self):
        """HTTP 429 is captured in errors list when raise_on_error=False."""
        mock_resp = _mock_http_response(429, text="Too Many Requests")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        collector = BlueskyCollector(session=mock_session, max_retries=1, retry_backoff=0.01)

        result = collector.collect("rate limit test", raise_on_error=False)
        self.assertEqual(result.total_records, 0)
        self.assertTrue(len(result.errors) > 0)

    def test_http_429_retry_count(self):
        """HTTP 429 with max_retries=1 results in exactly 2 HTTP calls."""
        mock_resp = _mock_http_response(429, text="Too Many Requests")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        collector = BlueskyCollector(session=mock_session, max_retries=1, retry_backoff=0.01)

        with self.assertRaises(BlueskyRateLimitError):
            collector._execute_request({"q": "test", "limit": 10, "sort": "latest"})

        self.assertEqual(mock_session.get.call_count, 2)  # initial + 1 retry

    def test_http_500_raises_api_error(self):
        """Non-200/403/429 status raises BlueskyAPIError."""
        mock_resp = _mock_http_response(500, text="Internal Server Error")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp
        collector = BlueskyCollector(session=mock_session, max_retries=0, retry_backoff=0.01)

        with self.assertRaises(BlueskyAPIError) as ctx:
            collector._execute_request({"q": "test", "limit": 10, "sort": "latest"})
        self.assertEqual(ctx.exception.status_code, 500)

    def test_timeout_raises_bluesky_timeout_error(self):
        """Network timeout raises BlueskyTimeoutError after retries."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")
        collector = BlueskyCollector(session=mock_session, max_retries=1, retry_backoff=0.01)

        with self.assertRaises(BlueskyTimeoutError):
            collector.collect("timeout test", raise_on_error=True)

    def test_network_error_captured_gracefully(self):
        """Network error captured in errors list when raise_on_error=False."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.ConnectionError("DNS failure")
        collector = BlueskyCollector(session=mock_session, max_retries=0, retry_backoff=0.01)

        result = collector.collect("network error test", raise_on_error=False)
        self.assertEqual(result.total_records, 0)
        self.assertTrue(len(result.errors) > 0)


# ===========================================================================
# 5. Schema Compatibility
# ===========================================================================
class TestBlueskySchemaCompatibility(unittest.TestCase):

    def test_bluesky_record_matches_gdelt_schema(self):
        """BlueskyRecord must have identical fields to GDELTArticle."""
        bsky_fields = set(BlueskyRecord.model_fields.keys())
        gdelt_fields = set(GDELTArticle.model_fields.keys())
        self.assertEqual(
            bsky_fields,
            gdelt_fields,
            f"Schema mismatch between BlueskyRecord and GDELTArticle! "
            f"Difference: {bsky_fields ^ gdelt_fields}",
        )

    def test_bluesky_record_matches_hackernews_schema(self):
        """BlueskyRecord must have identical fields to HackerNewsRecord."""
        bsky_fields = set(BlueskyRecord.model_fields.keys())
        hn_fields = set(HackerNewsRecord.model_fields.keys())
        self.assertEqual(
            bsky_fields,
            hn_fields,
            f"Schema mismatch between BlueskyRecord and HackerNewsRecord! "
            f"Difference: {bsky_fields ^ hn_fields}",
        )

    def test_bluesky_source_field_value(self):
        """source field must always be 'bluesky'."""
        post = _make_post()
        collector = BlueskyCollector(timeout=5.0)
        rec = collector._normalize_post(post, "test", "2026-11-18T00:00:00+00:00")
        self.assertEqual(rec.source, "bluesky")


# ===========================================================================
# 6. GDELT 429 Fix Tests
# ===========================================================================
class TestGDELT429Fixes(unittest.TestCase):
    """
    Focused tests for the GDELT 429 rate-limit fixes.

    Covers:
    - retry_backoff is always > min_request_interval (no double-sleep)
    - 429 body phrase detection (HTTP 200 with rate-limit body)
    - Hard cap: max_retries is clamped to 1 (2 total attempts)
    - _throttle() slot reservation prevents concurrent over-firing
    - Exactly 2 HTTP calls on 429 with max_retries=1
    """

    def setUp(self):
        from src.collectors.news.gdelt_collector import GDELTDocCollector
        self.GDELTDocCollector = GDELTDocCollector

    def test_retry_backoff_exceeds_min_interval(self):
        """retry_backoff must be strictly greater than min_request_interval."""
        collector = self.GDELTDocCollector(min_request_interval=6.0, retry_backoff=4.0)
        # Even if caller passes a backoff smaller than interval, it must be raised
        self.assertGreater(
            collector.retry_backoff,
            collector.min_request_interval,
            "retry_backoff must exceed min_request_interval to avoid double-sleep on retry",
        )

    def test_retry_backoff_default_exceeds_min_interval(self):
        """Default constructor must produce retry_backoff > min_request_interval."""
        collector = self.GDELTDocCollector()
        self.assertGreater(collector.retry_backoff, collector.min_request_interval)

    def test_max_retries_clamped_to_one(self):
        """max_retries is capped at 1 regardless of what caller passes."""
        collector_high = self.GDELTDocCollector(max_retries=10)
        self.assertEqual(collector_high.max_retries, 1)

        collector_zero = self.GDELTDocCollector(max_retries=0)
        self.assertEqual(collector_zero.max_retries, 0)

    def test_http_429_exactly_two_attempts(self):
        """HTTP 429 with max_retries=1 must make exactly 2 HTTP requests."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        # Use a MagicMock for .text so .lower() is callable on it
        mock_resp.text = MagicMock()
        mock_resp.text.__contains__ = lambda self, item: False  # no body phrase match
        mock_resp.text.lower.return_value = "rate limit exceeded"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        from src.collectors.news.gdelt_collector import GDELTRateLimitError
        collector = self.GDELTDocCollector(
            session=mock_session, max_retries=1, retry_backoff=0.01,
            min_request_interval=0.0,
        )

        with self.assertRaises(GDELTRateLimitError):
            collector._execute_request({"query": "test", "mode": "artlist", "format": "json"})

        self.assertEqual(mock_session.get.call_count, 2)

    def test_rate_limit_body_phrase_detected_on_http_200(self):
        """GDELT sometimes returns HTTP 200 with a rate-limit message in body."""
        from src.collectors.news.gdelt_collector import GDELTRateLimitError

        for phrase in [
            "Please limit requests to one every 5 seconds",
            "Rate Limit exceeded",
            "Too many requests",
            "slow down",
        ]:
            with self.subTest(phrase=phrase):
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.text = f"Error: {phrase}. Try again later."
                mock_session = MagicMock()
                mock_session.get.return_value = mock_resp

                collector = self.GDELTDocCollector(
                    session=mock_session,
                    max_retries=0,  # 0 retries → raises immediately
                    retry_backoff=0.01,
                    min_request_interval=0.0,
                )
                with self.assertRaises(GDELTRateLimitError):
                    collector._execute_request({"query": "test", "mode": "artlist", "format": "json"})

    def test_throttle_reserves_slot_before_sleeping(self):
        """
        After _throttle() returns, the shared _last_request_time must reflect
        the *scheduled departure time*, not the pre-sleep 'now'. This ensures
        a second concurrent call does not receive the same slot.
        """
        import time
        from src.collectors.news.gdelt_collector import GDELTDocCollector

        # Reset shared state for a clean test
        GDELTDocCollector._last_request_time = 0.0

        collector = GDELTDocCollector(min_request_interval=0.5)
        t_before = time.time()
        collector._throttle()  # first call: no sleep needed (last_time=0)
        reserved_time = GDELTDocCollector._last_request_time

        # The reserved slot should be at least t_before (no sleep on first call)
        self.assertGreaterEqual(reserved_time, t_before)

        # Second call should see that the slot is reserved and sleep ~0.5s
        t2_before = time.time()
        collector._throttle()
        t2_after = time.time()
        elapsed = t2_after - t2_before
        # Should have slept roughly 0.5s minus time already elapsed
        # We just check it did not fire immediately (< 0.1s would indicate no throttle)
        # Allow wide tolerance for CI slowness; the key invariant is > 0
        self.assertGreaterEqual(
            GDELTDocCollector._last_request_time,
            reserved_time + 0.4,  # slot moved forward by ~interval
        )

    def test_429_captured_gracefully(self):
        """HTTP 429 is captured in result.errors when raise_on_error=False."""
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate limit"
        mock_session = MagicMock()
        mock_session.get.return_value = mock_resp

        collector = self.GDELTDocCollector(
            session=mock_session, max_retries=1, retry_backoff=0.01,
            min_request_interval=0.0,
        )
        result = collector.collect("test query", raise_on_error=False)
        self.assertEqual(result.total_records, 0)
        self.assertTrue(len(result.errors) > 0)
        self.assertTrue(
            any("rate" in e.lower() or "429" in e or "limit" in e.lower() for e in result.errors)
        )


# ===========================================================================
# 7. Optional Live API Integration Test
# ===========================================================================
_RUN_LIVE = os.environ.get("RUN_BSKY_LIVE_TEST") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not _RUN_LIVE, reason="Live Bluesky API tests skipped by default. Set RUN_BSKY_LIVE_TEST=1 to run.")
def test_live_bluesky_api():
    """
    Live integration test querying the public Bluesky XRPC search endpoint.
    Only executed when RUN_BSKY_LIVE_TEST=1 is explicitly set.
    """
    collector = BlueskyCollector(timeout=15.0)
    result = collector.collect(
        query="open source intelligence",
        limit=5,
        sort="latest",
        raise_on_error=False,
    )

    print(f"\n[Bluesky Live Test] Query: 'open source intelligence'")
    print(f"[Bluesky Live Test] Total Records: {result.total_records}")
    print(f"[Bluesky Live Test] Errors: {result.errors}")

    # Live endpoint may return 0 results for obscure queries; that's valid behaviour
    assert isinstance(result, BlueskyQueryResult)
    assert result.query == "open source intelligence"

    if result.total_records > 0:
        first = result.records[0]
        assert first.source == "bluesky"
        assert first.id.startswith("bsky_")
        assert first.url.startswith("https://bsky.app/profile/")
        assert first.retrieved_at is not None
        assert "provenance" in first.metadata
        assert first.metadata["provenance"]["collector"] == "BlueskyCollector"


if __name__ == "__main__":
    if "--live" in sys.argv:
        os.environ["RUN_BSKY_LIVE_TEST"] = "1"
        print("Running live Bluesky API integration test…")
        test_live_bluesky_api()
        print("Live test completed successfully.")
    else:
        print("Running offline unit tests for Bluesky collector…")
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in (
            TestBlueskyUtilityFunctions,
            TestBlueskyCollectorParams,
            TestBlueskyPostNormalisation,
            TestBlueskyCollectorPipeline,
            TestBlueskySchemaCompatibility,
            TestGDELT429Fixes,
        ):
            suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=2)
        res = runner.run(suite)
        if not res.wasSuccessful():
            sys.exit(1)
