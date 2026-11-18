"""
GhostSignal — Discussion Collectors Module
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("HackerNewsCollector", "HackerNewsRecord", "HackerNewsQueryResult", "clean_html_text"):
        from src.collectors.discussions.hackernews_collector import (
            HackerNewsCollector,
            HackerNewsQueryResult,
            HackerNewsRecord,
            clean_html_text,
        )
        mapping = {
            "HackerNewsCollector": HackerNewsCollector,
            "HackerNewsRecord": HackerNewsRecord,
            "HackerNewsQueryResult": HackerNewsQueryResult,
            "clean_html_text": clean_html_text,
        }
        return mapping[name]
    if name in ("BlueskyCollector", "BlueskyRecord", "BlueskyQueryResult", "extract_external_links"):
        from src.collectors.discussions.bluesky_collector import (
            BlueskyCollector,
            BlueskyQueryResult,
            BlueskyRecord,
            extract_external_links,
        )
        mapping = {
            "BlueskyCollector": BlueskyCollector,
            "BlueskyRecord": BlueskyRecord,
            "BlueskyQueryResult": BlueskyQueryResult,
            "extract_external_links": extract_external_links,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HackerNewsCollector",
    "HackerNewsRecord",
    "HackerNewsQueryResult",
    "clean_html_text",
    "BlueskyCollector",
    "BlueskyRecord",
    "BlueskyQueryResult",
    "extract_external_links",
]
