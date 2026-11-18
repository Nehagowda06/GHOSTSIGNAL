"""
GhostSignal — News Collectors Module
"""

from src.collectors.news.gdelt_collector import (
    GDELTArticle,
    GDELTDocCollector,
    GDELTQueryResult,
    canonicalize_url,
)

__all__ = [
    "GDELTDocCollector",
    "GDELTArticle",
    "GDELTQueryResult",
    "canonicalize_url",
]
