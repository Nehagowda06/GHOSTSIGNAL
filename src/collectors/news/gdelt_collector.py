"""
GhostSignal — GDELT DOC API News Collector
===========================================
Collects news articles and metadata from the free GDELT DOC 2.0 API.

Endpoint:
    https://api.gdeltproject.org/api/v2/doc/doc

Data Contract Compliance:
    Output adheres to the shared GHOSTSIGNAL record schema defined in
    GHOSTSIGNAL_PROJECT_BRIEF.md (Section 6). Missing or unavailable
    fields (like full body, engagement metrics, author) are left as null/None.
    Article content is never fabricated; GDELT DOC API artlist mode provides
    titles and URLs, not full article bodies.

Rate Limit Note:
    The free GDELT DOC API enforces a strict rate limit of at most 1 request
    every 5 seconds. This collector automatically throttles requests with a
    configurable cooldown (default 6.0 s) and implements retry with backoff on
    HTTP 429.

    Root causes of repeated 429s — and how they are fixed here:
    ------------------------------------------------------------
    1. Concurrent requests: previously GDELT ran in a ThreadPoolExecutor
       alongside HN, allowing two requests to fire within the same 5-second
       window. Fixed by running GDELT sequentially *after* the concurrent
       HN + Bluesky step in investigation_service.py.

    2. Insufficient backoff after a 429: the previous backoff (6.0 s) equalled
       min_request_interval, so after sleeping the _throttle() call saw elapsed
       ≈ 0 s and slept *again*, effectively doubling the wait — but on the
       *second* retry the timestamps could race. Now the 429 backoff is
       explicitly set to max(retry_backoff, min_request_interval + 2.0) so
       at least 8 s pass between the failed request and the retry.

    3. Per-investigation attempt cap: GDELTDocCollector.max_retries is clamped
       to 1 (2 total attempts). The investigation service also stops calling
       GDELT after _GDELT_MAX_FAILURES (2) rate-limit errors per investigation,
       preventing repeated 429 floods across multiple collect() calls.

    4. 429 detection breadth: in addition to HTTP status 429, several known
       GDELT body phrases now trigger the rate-limit path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import threading
import time
import urllib.parse
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

import requests
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("ghostsignal.collectors.gdelt")
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
class GDELTCollectorError(Exception):
    """Base exception for GDELT collector errors."""
    pass


class GDELTRateLimitError(GDELTCollectorError):
    """Raised when GDELT rate limit (429 or 5-second interval) is exceeded."""
    pass


class GDELTTimeoutError(GDELTCollectorError):
    """Raised when request to GDELT API times out."""
    pass


class GDELTAPIError(GDELTCollectorError):
    """Raised when GDELT API returns an unhandled HTTP error or malformed payload."""
    def __init__(self, message: str, status_code: Optional[int] = None, response_text: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


# ---------------------------------------------------------------------------
# Data Models (GHOSTSIGNAL Section 6 Contract & FastAPI / Pydantic Compatible)
# ---------------------------------------------------------------------------
class GDELTArticle(BaseModel):
    """
    Normalized GHOSTSIGNAL record for an article retrieved from GDELT.
    Adheres strictly to GHOSTSIGNAL_PROJECT_BRIEF.md Data Contract.
    """
    id: str = Field(..., description="Stable unique record identifier")
    source: str = Field(default="gdelt", description="Canonical source identifier ('gdelt')")
    content: str = Field(
        ...,
        description="Main text for analysis (Article title as GDELT artlist does not provide full bodies; never fabricated)"
    )
    title: Optional[str] = Field(default=None, description="Article title")
    url: Optional[str] = Field(default=None, description="Original source URL")
    published_at: Optional[str] = Field(
        default=None,
        description="ISO-8601 publication date from GDELT seendate, if parseable"
    )
    retrieved_at: str = Field(..., description="ISO-8601 UTC timestamp of retrieval")
    author: Optional[str] = Field(default=None, description="Author if available (None for GDELT artlist)")
    product: Optional[str] = Field(default=None, description="Product identifier (None for news)")
    rating: Optional[float] = Field(default=None, description="Rating (None for news)")
    likes: Optional[int] = Field(default=None, description="Likes count (None for news)")
    comments: Optional[int] = Field(default=None, description="Comments count (None for news)")
    shares: Optional[int] = Field(default=None, description="Shares count (None for news)")
    views: Optional[int] = Field(default=None, description="Views count (None for news)")
    label: Optional[str] = Field(default=None, description="Ground truth training label")
    predicted_label: Optional[str] = Field(default=None, description="Model prediction")
    confidence: Optional[float] = Field(default=None, description="Model prediction confidence")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata and provenance information"
    )


class GDELTQueryResult(BaseModel):
    """Container for collector execution output and diagnostics."""
    query: str
    total_records: int
    retrieved_at: str
    articles: List[GDELTArticle] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to standard dictionary."""
        return self.model_dump()

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize result to structured JSON string."""
        return self.model_dump_json(indent=indent)

    def to_records(self) -> List[Dict[str, Any]]:
        """Return articles as a list of dictionaries matching GHOSTSIGNAL schema."""
        return [art.model_dump() for art in self.articles]


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
# Tracking parameters routinely appended to news URLs
_TRACKING_QUERY_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "fbclid", "gclid", "msclkid", "ref", "ref_src", "source",
    "mc_cid", "mc_eid", "igshid", "ocid", "cmpid", "mbid", "taid"
})


def canonicalize_url(url: Optional[str]) -> str:
    """
    Produce a canonical, normalized URL string suitable for deduplication.

    Transformations:
    - Strips whitespace.
    - Lowercases scheme and network location (hostname).
    - Removes standard default ports (:80, :443).
    - Cleans duplicate path slashes and trailing slashes.
    - Strips tracking query parameters (e.g. utm_*, fbclid, gclid).
    - Sorts remaining query parameters deterministically.
    - Removes fragments (#...).
    """
    if not url or not isinstance(url, str):
        return ""

    url_str = url.strip()
    if not url_str:
        return ""

    try:
        parsed = urllib.parse.urlparse(url_str)
        scheme = (parsed.scheme or "http").lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if netloc.endswith(":80") and scheme == "http":
            netloc = netloc[:-3]
        elif netloc.endswith(":443") and scheme == "https":
            netloc = netloc[:-4]

        # Strip optional leading 'www.' for host deduplication
        # (keeps subdomain handling consistent)
        if netloc.startswith("www."):
            netloc = netloc[4:]

        # Normalize path
        path = parsed.path or "/"
        while "//" in path:
            path = path.replace("//", "/")
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]

        # Filter out tracking query parameters
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
        clean_pairs = [
            (k, v) for k, v in query_pairs if k.lower() not in _TRACKING_QUERY_PARAMS
        ]
        clean_pairs.sort(key=lambda item: (item[0], item[1]))
        clean_query = urllib.parse.urlencode(clean_pairs)

        return urllib.parse.urlunparse((scheme, netloc, path, "", clean_query, ""))
    except Exception as e:
        log.debug("URL canonicalization failed for '%s': %s", url_str, e)
        return url_str


def generate_record_id(canonical_url: str, title: Optional[str] = None) -> str:
    """Generate a stable, unique 16-character identifier for the record."""
    basis = canonical_url if canonical_url else (title or "unknown")
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]
    return f"gdelt_{digest}"


def parse_gdelt_datetime(date_str: Optional[str]) -> Optional[str]:
    """
    Parse GDELT's seendate string into a standardized ISO-8601 UTC string.

    GDELT typically returns:
      - '20261025T033000Z'
      - '20261025033000'
      - '20261025'
    """
    if not date_str or not isinstance(date_str, str):
        return None

    cleaned = date_str.strip()
    if not cleaned:
        return None

    formats = [
        "%Y%m%dT%H%M%SZ",
        "%Y%m%d%H%M%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y%m%d",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue

    log.debug("Unrecognized GDELT seendate format: %s", cleaned)
    return None


def format_gdelt_datetime(dt_val: Union[datetime, date, str]) -> str:
    """
    Format a datetime, date, or date string into GDELT's YYYYMMDDHHMMSS format.
    """
    if isinstance(dt_val, datetime):
        return dt_val.strftime("%Y%m%d%H%M%S")
    if isinstance(dt_val, date):
        return dt_val.strftime("%Y%m%d") + "000000"

    # String format parsing
    val = dt_val.strip()
    # If already 14 digits
    if re.fullmatch(r"\d{14}", val):
        return val
    # If 8 digits YYYYMMDD
    if re.fullmatch(r"\d{8}", val):
        return val + "000000"

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            parsed = datetime.strptime(val, fmt)
            return parsed.strftime("%Y%m%d%H%M%S")
        except ValueError:
            pass

    raise ValueError(f"Cannot format date '{dt_val}' for GDELT API. Expected datetime, date, or YYYY-MM-DD string.")


# ---------------------------------------------------------------------------
# GDELT DOC API Collector
# ---------------------------------------------------------------------------
class GDELTDocCollector:
    """
    News Collector targeting the free GDELT DOC 2.0 API.

    Features:
    - User-provided investigation queries with optional domain/country/language filters.
    - Deduplication using canonical URLs.
    - Configurable date ranges (start_date, end_date) or timespans ('24h', '7d', etc.).
    - Strict rate-limiting compliance (inter-request cooldown, default 5.5s).
    - Exponential backoff retry on HTTP 429 rate limit errors.
    - Adherence to GHOSTSIGNAL common schema without data fabrication.
    """

    ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
    DEFAULT_USER_AGENT = "GhostSignal-IntelligenceCollector/1.0 (+https://ghostsignal.local)"

    # Shared class-level lock and timestamp to coordinate requests across instances/threads.
    # Using a process-wide lock ensures that even if multiple GDELTDocCollector instances
    # exist they still respect the shared 5-second rate-limit window.
    _rate_lock = threading.Lock()
    _last_request_time: float = 0.0

    # Known rate-limit body phrases returned by GDELT (in addition to HTTP 429)
    _RATE_LIMIT_PHRASES = (
        "Please limit requests to one every 5 seconds",
        "rate limit",
        "too many requests",
        "slow down",
    )

    def __init__(
        self,
        endpoint: str = ENDPOINT,
        timeout: float = 15.0,
        min_request_interval: float = 6.0,
        max_retries: int = 1,
        retry_backoff: float = 8.0,
        user_agent: str = DEFAULT_USER_AGENT,
        session: Optional[requests.Session] = None,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.min_request_interval = min_request_interval
        # Hard-cap at 1 retry → 2 total attempts per collect() call.
        # The investigation service enforces an additional per-investigation cap.
        self.max_retries = max(0, min(max_retries, 1))
        # On a 429 the backoff must exceed min_request_interval so that _throttle()
        # on the retry does not add another full interval on top (double-sleep bug).
        # We ensure at least min_request_interval + 2 s of real wall-clock separation.
        self.retry_backoff = max(retry_backoff, self.min_request_interval + 2.0)
        self.user_agent = user_agent
        self.session = session or requests.Session()

    def _throttle(self) -> None:
        """
        Enforce the GDELT minimum inter-request interval across all calls in this process.

        The lock is held only while measuring and updating the shared timestamp;
        the sleep happens *outside* the lock so other threads are not blocked
        for the full sleep duration.
        """
        with self._rate_lock:
            now = time.time()
            elapsed = now - GDELTDocCollector._last_request_time
            sleep_duration = self.min_request_interval - elapsed if elapsed < self.min_request_interval else 0.0
            # Reserve the slot immediately so a concurrent caller sees an updated timestamp
            GDELTDocCollector._last_request_time = now + sleep_duration

        if sleep_duration > 0:
            log.debug("Throttling GDELT request: sleeping %.2f seconds", sleep_duration)
            time.sleep(sleep_duration)

    def build_query_string(
        self,
        query: str,
        domain: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> str:
        """
        Compose the GDELT search query with optional domain, sourcelang, or sourcecountry filters.
        """
        parts = [query.strip()]
        if domain:
            clean_dom = domain.strip().lower()
            if clean_dom.startswith("www."):
                clean_dom = clean_dom[4:]
            parts.append(f"domain:{clean_dom}")
        if language:
            parts.append(f"sourcelang:{language.strip().lower()}")
        if country:
            parts.append(f"sourcecountry:{country.strip().upper()}")

        return " ".join(parts)

    def build_params(
        self,
        query: str,
        max_records: int = 75,
        start_date: Optional[Union[datetime, date, str]] = None,
        end_date: Optional[Union[datetime, date, str]] = None,
        timespan: Optional[str] = None,
        sort: str = "DateDesc",
        domain: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build and validate query parameters for the GDELT DOC API.
        """
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        # GDELT DOC API allows max 250 records per request
        clamped_records = max(1, min(int(max_records), 250))
        full_query = self.build_query_string(query, domain=domain, language=language, country=country)

        params: Dict[str, Any] = {
            "query": full_query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": clamped_records,
            "sort": sort,
        }

        # Date range parameters
        if timespan:
            params["timespan"] = timespan.strip()
        else:
            if start_date:
                params["startdatetime"] = format_gdelt_datetime(start_date)
            if end_date:
                params["enddatetime"] = format_gdelt_datetime(end_date)

        return params

    def _execute_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the HTTP request with throttling, error inspection, and retry on 429.
        """
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/plain, */*",
        }

        attempts = 0
        while attempts <= self.max_retries:
            attempts += 1
            self._throttle()

            log.info("Sending request to GDELT DOC API (attempt %d/%d)", attempts, self.max_retries + 1)
            try:
                response = self.session.get(
                    self.endpoint,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.exceptions.Timeout as e:
                log.warning("GDELT API request timed out: %s", e)
                if attempts > self.max_retries:
                    raise GDELTTimeoutError(f"GDELT DOC API timed out after {self.timeout}s: {e}") from e
                time.sleep(self.retry_backoff * attempts)
                continue
            except requests.exceptions.RequestException as e:
                log.error("Network error during GDELT request: %s", e)
                if attempts > self.max_retries:
                    raise GDELTCollectorError(f"Network error querying GDELT API: {e}") from e
                time.sleep(self.retry_backoff * attempts)
                continue

            # Check for Rate Limiting (HTTP 429 or known rate-limit body phrases).
            # GDELT sometimes returns HTTP 200 with a plain-text rate-limit message,
            # so we also check the response body against several known phrases.
            response_body_lower = response.text.lower()
            is_rate_limited = response.status_code == 429 or any(
                phrase.lower() in response_body_lower
                for phrase in self._RATE_LIMIT_PHRASES
            )
            if is_rate_limited:
                # Use a fixed backoff that is strictly larger than min_request_interval
                # so the next _throttle() call does not add a redundant sleep on top.
                wait_time = self.retry_backoff
                log.warning(
                    "GDELT Rate Limit hit (HTTP %s). Backing off for %.1fs (attempt %d/%d)",
                    response.status_code,
                    wait_time,
                    attempts,
                    self.max_retries + 1,
                )
                if attempts > self.max_retries:
                    raise GDELTRateLimitError(
                        f"GDELT rate limit exceeded after {attempts} attempt(s). "
                        f"Response: {response.text[:200]}"
                    )
                time.sleep(wait_time)
                # Reset the shared timestamp so _throttle() on the next attempt
                # does not add another full interval on top of the backoff we just slept.
                with self._rate_lock:
                    GDELTDocCollector._last_request_time = time.time()
                continue

            # Check HTTP status
            if response.status_code != 200:
                raise GDELTAPIError(
                    f"GDELT API returned HTTP status {response.status_code}",
                    status_code=response.status_code,
                    response_text=response.text[:500],
                )

            # Handle response parsing
            body = response.text.strip()
            if not body:
                log.info("GDELT returned empty response body (no records found).")
                return {"articles": []}

            # GDELT occasionally returns plain HTML when zero results match
            if body.startswith("<") or "Your search did not match any documents" in body:
                log.info("GDELT returned search miss notification HTML.")
                return {"articles": []}

            try:
                data = response.json()
                if not isinstance(data, dict):
                    log.warning("GDELT returned unexpected JSON type: %s", type(data))
                    return {"articles": []}
                return data
            except (requests.exceptions.JSONDecodeError, json.JSONDecodeError, ValueError) as e:
                log.warning("Failed to decode JSON from GDELT response: %s (Body snippet: %s)", e, body[:200])
                if "articles" not in body.lower():
                    # Empty or non-matching message
                    return {"articles": []}
                raise GDELTAPIError(
                    f"Malformed JSON from GDELT API: {e}",
                    status_code=response.status_code,
                    response_text=body[:500],
                ) from e

        raise GDELTCollectorError("Exhausted retries while requesting GDELT DOC API.")

    def collect(
        self,
        query: str,
        max_records: int = 75,
        start_date: Optional[Union[datetime, date, str]] = None,
        end_date: Optional[Union[datetime, date, str]] = None,
        timespan: Optional[str] = None,
        sort: str = "DateDesc",
        domain: Optional[str] = None,
        language: Optional[str] = None,
        country: Optional[str] = None,
        deduplicate: bool = True,
        raise_on_error: bool = False,
    ) -> GDELTQueryResult:
        """
        Execute an investigation query against GDELT DOC API and return normalized records.

        Parameters
        ----------
        query : str
            Investigation search query.
        max_records : int, optional
            Maximum records to retrieve (1 to 250, default 75).
        start_date : datetime | date | str, optional
            Start date boundary.
        end_date : datetime | date | str, optional
            End date boundary.
        timespan : str, optional
            Relative timespan (e.g. '15m', '24h', '7d', '1w', '1m', '3m', '1y').
        sort : str, optional
            Sort order ('DateDesc', 'DateAsc', 'ToneDesc', 'ToneAsc', 'Rel').
        domain : str, optional
            Filter to a specific domain (e.g. 'reuters.com').
        language : str, optional
            Filter to a language name/code (e.g. 'English', 'eng').
        country : str, optional
            Filter to source country (e.g. 'US').
        deduplicate : bool, optional
            Whether to deduplicate articles by canonical URL (default True).
        raise_on_error : bool, optional
            If False (default), returns a result container with errors list.
            If True, re-raises collector exceptions.

        Returns
        -------
        GDELTQueryResult
            Validated result containing articles list adhering to GHOSTSIGNAL schema.
        """
        retrieved_at_iso = datetime.now(timezone.utc).isoformat()
        errors: List[str] = []

        try:
            params = self.build_params(
                query=query,
                max_records=max_records,
                start_date=start_date,
                end_date=end_date,
                timespan=timespan,
                sort=sort,
                domain=domain,
                language=language,
                country=country,
            )
            raw_payload = self._execute_request(params)
        except Exception as e:
            log.error("GDELT collection failed for query '%s': %s", query, e)
            if raise_on_error:
                raise
            errors.append(str(e))
            return GDELTQueryResult(
                query=query,
                total_records=0,
                retrieved_at=retrieved_at_iso,
                articles=[],
                errors=errors,
            )

        raw_articles = raw_payload.get("articles") or []
        articles: List[GDELTArticle] = []
        seen_canonical_urls: Set[str] = set()

        for raw_art in raw_articles:
            if not isinstance(raw_art, dict):
                continue

            orig_url = raw_art.get("url")
            raw_title = raw_art.get("title")
            raw_seendate = raw_art.get("seendate")
            raw_domain = raw_art.get("domain") or ""
            raw_language = raw_art.get("language")
            raw_country = raw_art.get("sourcecountry")
            raw_image = raw_art.get("socialimage")
            raw_mobile = raw_art.get("url_mobile")

            # Canonicalize URL for deduplication and stable id
            canonical_url = canonicalize_url(orig_url)

            if deduplicate and canonical_url:
                if canonical_url in seen_canonical_urls:
                    log.debug("Deduplicated duplicate canonical URL: %s", canonical_url)
                    continue
                seen_canonical_urls.add(canonical_url)

            record_id = generate_record_id(canonical_url, raw_title)
            published_at = parse_gdelt_datetime(raw_seendate)

            # Preserve title as main analysis content (GDELT does not provide full bodies; no fabrication)
            content_text = (raw_title or "").strip()

            metadata: Dict[str, Any] = {
                "canonical_url": canonical_url,
                "domain": raw_domain,
                "language": raw_language,
                "source_country": raw_country,
                "social_image": raw_image or None,
                "mobile_url": raw_mobile or None,
                "seendate_raw": raw_seendate,
                "provenance": {
                    "collector": "GDELTDocCollector",
                    "api_endpoint": self.endpoint,
                    "query": query,
                    "retrieved_at": retrieved_at_iso,
                    "source_domain": raw_domain,
                    "source_country": raw_country,
                    "language": raw_language,
                    "original_url": orig_url,
                    "canonical_url": canonical_url,
                },
            }

            article = GDELTArticle(
                id=record_id,
                source="gdelt",
                content=content_text,
                title=raw_title,
                url=orig_url,
                published_at=published_at,
                retrieved_at=retrieved_at_iso,
                author=None,
                product=None,
                rating=None,
                likes=None,
                comments=None,
                shares=None,
                views=None,
                label=None,
                predicted_label=None,
                confidence=None,
                metadata=metadata,
            )
            articles.append(article)

        log.info(
            "GDELT collection complete for '%s': %d retrieved, %d after deduplication",
            query,
            len(raw_articles),
            len(articles),
        )

        return GDELTQueryResult(
            query=query,
            total_records=len(articles),
            retrieved_at=retrieved_at_iso,
            articles=articles,
            errors=errors,
        )

    def collect_records(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        """Convenience method returning records as standard dictionaries."""
        result = self.collect(*args, **kwargs)
        return result.to_records()

    def collect_json(self, *args: Any, indent: Optional[int] = 2, **kwargs: Any) -> str:
        """Convenience method returning JSON string output."""
        result = self.collect(*args, **kwargs)
        return result.to_json(indent=indent)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="GhostSignal GDELT DOC API News Collector CLI"
    )
    parser.add_argument("query", nargs="?", default="digital disinformation", help="Investigation query")
    parser.add_argument("--limit", type=int, default=5, help="Result limit (max 250)")
    parser.add_argument("--timespan", type=str, default="7d", help="Relative timespan (e.g. 24h, 7d, 1m)")
    parser.add_argument("--domain", type=str, default=None, help="Domain filter (e.g. bbc.com)")
    parser.add_argument("--language", type=str, default=None, help="Language filter")
    parser.add_argument("--country", type=str, default=None, help="Country filter (e.g. US)")
    parser.add_argument("--output", type=str, default=None, help="Path to write output JSON")

    args = parser.parse_args()

    collector = GDELTDocCollector()
    result = collector.collect(
        query=args.query,
        max_records=args.limit,
        timespan=args.timespan,
        domain=args.domain,
        language=args.language,
        country=args.country,
    )

    json_str = result.to_json(indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"Results written to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    _main()
