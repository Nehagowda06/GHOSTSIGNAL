"""
GhostSignal — Bluesky Public Discussion Collector
=================================================
Collects public posts, author metadata, engagement metrics, and external links
from the public Bluesky AT Protocol XRPC API.

Endpoint:
    https://api.bsky.app/xrpc/app.bsky.feed.searchPosts

Authentication:
    No API key or authentication required for the public search endpoint.

Data Contract Compliance:
    Output adheres strictly to the shared GHOSTSIGNAL unified schema
    (Section 6 of Project Brief). Post text, author handles, timestamps,
    and engagement counts are extracted faithfully from API responses
    and never fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

import requests
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("ghostsignal.collectors.bluesky")
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    )
    log.addHandler(handler)
    log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BlueskyCollectorError(Exception):
    """Base exception for Bluesky collector errors."""
    pass


class BlueskyForbiddenError(BlueskyCollectorError):
    """Raised when Bluesky returns HTTP 403 (e.g. Cloudflare / WAF block)."""
    pass


class BlueskyRateLimitError(BlueskyCollectorError):
    """Raised when Bluesky returns HTTP 429 (rate limited)."""
    pass


class BlueskyTimeoutError(BlueskyCollectorError):
    """Raised when a request to the Bluesky API times out."""
    pass


class BlueskyAPIError(BlueskyCollectorError):
    """Raised when Bluesky API returns an unexpected HTTP error or malformed payload."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


# ---------------------------------------------------------------------------
# Data Models (GHOSTSIGNAL Unified Schema)
# ---------------------------------------------------------------------------
class BlueskyRecord(BaseModel):
    """
    Normalized GHOSTSIGNAL record for a post retrieved from Bluesky.
    Compatible with GDELTArticle, HackerNewsRecord, and the GHOSTSIGNAL Data Contract.
    """
    id: str = Field(..., description="Stable unique record identifier (e.g. bsky_3n2ku...)")
    source: str = Field(default="bluesky", description="Canonical source identifier ('bluesky')")
    content: str = Field(
        ...,
        description="Main text for analysis (post body; never fabricated)"
    )
    title: Optional[str] = Field(default=None, description="Optional title or preview (None for microblog posts)")
    url: Optional[str] = Field(default=None, description="Direct web permalink to the post on bsky.app")
    published_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 publication date in UTC from post createdAt"
    )
    retrieved_at: str = Field(..., description="ISO-8601 UTC timestamp of retrieval")
    author: Optional[str] = Field(default=None, description="Author handle (e.g. user.bsky.social)")
    product: Optional[str] = Field(default=None, description="Product identifier (None for discussions)")
    rating: Optional[float] = Field(default=None, description="Rating (None for discussions)")
    likes: Optional[int] = Field(default=None, description="Like count if available")
    comments: Optional[int] = Field(default=None, description="Reply count if available")
    shares: Optional[int] = Field(default=None, description="Repost count if available")
    views: Optional[int] = Field(default=None, description="Views count (None for Bluesky)")
    label: Optional[str] = Field(default=None, description="Ground truth training label")
    predicted_label: Optional[str] = Field(default=None, description="Model prediction")
    confidence: Optional[float] = Field(default=None, description="Model prediction confidence")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata, handles, DIDs, external links, and provenance"
    )


class BlueskyQueryResult(BaseModel):
    """Container for Bluesky collector execution output and diagnostics."""
    query: str
    total_records: int
    retrieved_at: str
    records: List[BlueskyRecord] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to standard dictionary."""
        return self.model_dump()

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize result container to structured JSON string."""
        return self.model_dump_json(indent=indent)

    def to_records(self) -> List[Dict[str, Any]]:
        """Return items as a list of dictionaries matching the GHOSTSIGNAL unified schema."""
        return [rec.model_dump() for rec in self.records]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def parse_bsky_rkey(uri: Optional[str]) -> str:
    """
    Extract the record key (rkey) from an AT-URI.

    Example:
        'at://did:plc:wcgbb35w2zjdq4gg42qjqqn3/app.bsky.feed.post/3n2kuor6tsi26' -> '3n2kuor6tsi26'
    """
    if not uri or not isinstance(uri, str):
        return ""
    parts = uri.strip().rstrip("/").split("/")
    return parts[-1] if parts else ""


def build_bsky_permalink(handle_or_did: str, rkey: str) -> str:
    """
    Construct the public web URL for a Bluesky post.

    Example:
        'alice.bsky.social', '3n2kuor6tsi26' -> 'https://bsky.app/profile/alice.bsky.social/post/3n2kuor6tsi26'
    """
    author = handle_or_did.strip() if handle_or_did else "profile"
    post_key = rkey.strip()
    return f"https://bsky.app/profile/{author}/post/{post_key}"


def extract_external_links(post_data: Dict[str, Any]) -> List[str]:
    """
    Extract external URLs referenced in post embeds or rich-text facets.
    Never fabricates links.
    """
    links: List[str] = []
    rec = post_data.get("record") or {}
    embed = rec.get("embed") or {}

    # 1. External link card embed
    embed_type = embed.get("$type")
    if embed_type == "app.bsky.embed.external":
        ext_uri = embed.get("external", {}).get("uri")
        if ext_uri and isinstance(ext_uri, str) and ext_uri.strip():
            links.append(ext_uri.strip())
    elif embed_type == "app.bsky.embed.recordWithMedia":
        ext_uri = embed.get("media", {}).get("external", {}).get("uri")
        if ext_uri and isinstance(ext_uri, str) and ext_uri.strip():
            links.append(ext_uri.strip())

    # 2. Rich-text facets (inline links in post body)
    for facet in rec.get("facets") or []:
        for feature in facet.get("features") or []:
            if feature.get("$type") == "app.bsky.richtext.facet#link":
                uri = feature.get("uri")
                if uri and isinstance(uri, str) and uri.strip():
                    clean_u = uri.strip()
                    if clean_u not in links:
                        links.append(clean_u)

    return links


# ---------------------------------------------------------------------------
# Bluesky Collector
# ---------------------------------------------------------------------------
class BlueskyCollector:
    """
    Collector targeting the public Bluesky AT Protocol XRPC search endpoint.

    Features:
    - Search posts using user-provided investigation query.
    - Full schema compatibility with GHOSTSIGNAL data contract, GDELT, and Hacker News.
    - Preserves author handle, display name, DID, engagement metrics, and external URLs.
    - Resilient failure handling: gracefully absorbs HTTP 403, 429, timeouts, and network issues.
    - Never fabricates content, metrics, or timestamps.
    """

    SEARCH_ENDPOINT = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
    DEFAULT_USER_AGENT = "GhostSignal-BlueskyCollector/1.0 (+https://ghostsignal.local)"

    def __init__(
        self,
        endpoint: str = SEARCH_ENDPOINT,
        timeout: float = 12.0,
        max_retries: int = 1,
        retry_backoff: float = 2.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: Optional[requests.Session] = None,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.user_agent = user_agent
        self.session = session or requests.Session()

    def build_params(
        self,
        query: str,
        limit: int = 50,
        sort: str = "latest",
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Construct query parameters for Bluesky searchPosts XRPC endpoint."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        clamped_limit = max(1, min(int(limit), 100))
        params: Dict[str, Any] = {
            "q": query.strip(),
            "limit": clamped_limit,
            "sort": sort,
        }
        if since:
            params["since"] = since.strip()
        if until:
            params["until"] = until.strip()

        return params

    def _execute_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute HTTP request to Bluesky XRPC with error inspection and graceful exception mapping.
        """
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        attempts = 0
        while attempts <= self.max_retries:
            attempts += 1
            log.debug("Querying Bluesky XRPC API (attempt %d/%d): %s", attempts, self.max_retries + 1, params)
            try:
                response = self.session.get(
                    self.endpoint,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as e:
                log.warning("Bluesky API request timed out: %s", e)
                if attempts > self.max_retries:
                    raise BlueskyTimeoutError(f"Bluesky API timed out after {self.timeout}s: {e}") from e
                time.sleep(self.retry_backoff * attempts)
                continue
            except requests.exceptions.RequestException as e:
                log.error("Network error during Bluesky API request: %s", e)
                if attempts > self.max_retries:
                    raise BlueskyCollectorError(f"Network error querying Bluesky API: {e}") from e
                time.sleep(self.retry_backoff * attempts)
                continue

            # Check status codes
            if response.status_code == 403:
                raise BlueskyForbiddenError(
                    f"Bluesky API access forbidden (HTTP 403): {response.text[:200]}"
                )

            if response.status_code == 429:
                wait_time = self.retry_backoff * attempts
                log.warning("Bluesky rate limited (HTTP 429). Backing off for %.1fs", wait_time)
                if attempts > self.max_retries:
                    raise BlueskyRateLimitError(
                        f"Bluesky rate limit exceeded (HTTP 429): {response.text[:200]}"
                    )
                time.sleep(wait_time)
                continue

            if response.status_code != 200:
                raise BlueskyAPIError(
                    f"Bluesky API error HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                    response_text=response.text[:500],
                )

            body = response.text.strip()
            if not body:
                return {"posts": [], "hitsTotal": 0}

            try:
                data = response.json()
                if not isinstance(data, dict):
                    log.warning("Bluesky returned unexpected JSON type: %s", type(data))
                    return {"posts": [], "hitsTotal": 0}
                return data
            except (requests.exceptions.JSONDecodeError, json.JSONDecodeError, ValueError) as e:
                raise BlueskyAPIError(
                    f"Malformed JSON from Bluesky API: {e}",
                    status_code=response.status_code,
                    response_text=body[:500],
                ) from e

        raise BlueskyCollectorError("Exhausted retries while querying Bluesky API.")

    def _normalize_post(
        self,
        post_data: Dict[str, Any],
        query: str,
        retrieved_at_iso: str,
    ) -> Optional[BlueskyRecord]:
        """Transform raw Bluesky XRPC post into a normalized GHOSTSIGNAL record."""
        if not isinstance(post_data, dict):
            return None

        raw_uri = post_data.get("uri")
        rkey = parse_bsky_rkey(raw_uri)
        if not rkey:
            # Fallback to cid or hash of URI
            cid = post_data.get("cid") or ""
            rkey = cid[:16] if cid else hashlib.sha256(str(raw_uri).encode("utf-8")).hexdigest()[:16]

        record_id = f"bsky_{rkey}"
        rec = post_data.get("record") or {}
        text = rec.get("text") or ""
        content_text = text.strip() if isinstance(text, str) else ""

        author_info = post_data.get("author") or {}
        handle = author_info.get("handle") or ""
        did = author_info.get("did") or ""
        display_name = author_info.get("displayName") or None

        # Build direct web URL to post
        url = build_bsky_permalink(handle or did, rkey)

        # Timestamps
        created_at = rec.get("createdAt") or post_data.get("indexedAt")
        published_at = created_at if isinstance(created_at, str) and created_at.strip() else None

        # Engagement metrics
        likes = post_data.get("likeCount")
        replies = post_data.get("replyCount")
        reposts = post_data.get("repostCount")

        likes_val = int(likes) if isinstance(likes, (int, float)) else None
        replies_val = int(replies) if isinstance(replies, (int, float)) else None
        reposts_val = int(reposts) if isinstance(reposts, (int, float)) else None

        external_links = extract_external_links(post_data)

        metadata: Dict[str, Any] = {
            "item_type": "post",
            "uri": raw_uri,
            "cid": post_data.get("cid"),
            "did": did,
            "handle": handle,
            "display_name": display_name,
            "quote_count": post_data.get("quoteCount"),
            "bookmark_count": post_data.get("bookmarkCount"),
            "external_links": external_links,
            "provenance": {
                "collector": "BlueskyCollector",
                "api_endpoint": self.endpoint,
                "query": query,
                "retrieved_at": retrieved_at_iso,
                "uri": raw_uri,
                "rkey": rkey,
            },
        }

        return BlueskyRecord(
            id=record_id,
            source="bluesky",
            content=content_text,
            title=None,
            url=url,
            published_at=published_at,
            retrieved_at=retrieved_at_iso,
            author=handle or display_name or did or None,
            product=None,
            rating=None,
            likes=likes_val,
            comments=replies_val,
            shares=reposts_val,
            views=None,
            label=None,
            predicted_label=None,
            confidence=None,
            metadata=metadata,
        )

    def collect(
        self,
        query: str,
        limit: int = 50,
        sort: str = "latest",
        since: Optional[str] = None,
        until: Optional[str] = None,
        deduplicate: bool = True,
        raise_on_error: bool = False,
    ) -> BlueskyQueryResult:
        """
        Execute an investigation search query against Bluesky and return normalized records.

        Parameters
        ----------
        query : str
            Investigation search query.
        limit : int, optional
            Maximum records to retrieve (1 to 100, default 50).
        sort : str, optional
            Sort order ('latest' or 'top', default 'latest').
        since : str, optional
            ISO-8601 start date boundary.
        until : str, optional
            ISO-8601 end date boundary.
        deduplicate : bool, optional
            Whether to deduplicate posts by record ID (default True).
        raise_on_error : bool, optional
            If False (default), catches errors and returns structured result with errors list.
            If True, re-raises collector exceptions.

        Returns
        -------
        BlueskyQueryResult
            Structured result containing validated records adhering to GHOSTSIGNAL schema.
        """
        retrieved_at_iso = datetime.now(timezone.utc).isoformat()
        errors: List[str] = []
        records: List[BlueskyRecord] = []
        seen_ids: Set[str] = set()

        try:
            params = self.build_params(query=query, limit=limit, sort=sort, since=since, until=until)
            raw_payload = self._execute_request(params)
        except Exception as e:
            log.warning("Bluesky collection failed for query '%s': %s", query, e)
            if raise_on_error:
                raise
            errors.append(str(e))
            return BlueskyQueryResult(
                query=query,
                total_records=0,
                retrieved_at=retrieved_at_iso,
                records=[],
                errors=errors,
            )

        raw_posts = raw_payload.get("posts") or []
        for raw_post in raw_posts:
            rec = self._normalize_post(raw_post, query, retrieved_at_iso)
            if rec is None:
                continue

            if deduplicate and rec.id in seen_ids:
                log.debug("Skipping duplicate Bluesky post: %s", rec.id)
                continue

            records.append(rec)
            seen_ids.add(rec.id)

        log.info(
            "Bluesky collection finished for '%s': %d retrieved, %d after deduplication",
            query,
            len(raw_posts),
            len(records),
        )

        return BlueskyQueryResult(
            query=query,
            total_records=len(records),
            retrieved_at=retrieved_at_iso,
            records=records,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    """Command-line interface for running the Bluesky collector independently."""
    parser = argparse.ArgumentParser(
        description="GhostSignal Bluesky Discussion Collector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="Search query to investigate on Bluesky")
    parser.add_argument("--limit", type=int, default=25, help="Maximum posts to retrieve (1-100)")
    parser.add_argument("--sort", type=str, default="latest", choices=["latest", "top"], help="Sort order")
    parser.add_argument("--json", action="store_true", help="Output full JSON instead of human summary")
    args = parser.parse_args()

    collector = BlueskyCollector()
    result = collector.collect(query=args.query, limit=args.limit, sort=args.sort)

    if args.json:
        print(result.to_json(indent=2))
    else:
        print(f"\nBluesky Investigation: '{result.query}'")
        print(f"Total posts: {result.total_records}")
        print(f"Retrieved at: {result.retrieved_at}")
        if result.errors:
            print(f"Errors: {'; '.join(result.errors)}")
        print("\nSample Posts:")
        for i, post in enumerate(result.records[:5], 1):
            print(f"  [{i}] @{post.author} ({post.published_at or 'undated'})")
            print(f"      {post.content[:120]}...")
            print(f"      URL: {post.url} | Likes: {post.likes} | Replies: {post.comments} | Reposts: {post.shares}")
            links = post.metadata.get("external_links", [])
            if links:
                print(f"      External links: {', '.join(links)}")
            print()


if __name__ == "__main__":
    main()
