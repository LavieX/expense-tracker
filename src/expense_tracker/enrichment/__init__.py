"""Enrichment module for Expense Tracker.

Enrichment providers scrape external data sources (e.g. Amazon order history)
and produce cache files that the pipeline's enrich stage consumes to split
aggregate transactions into individual line items.

The module defines an :class:`EnrichmentProvider` protocol and a registry
for looking up providers by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from expense_tracker.enrichment.cache import EnrichmentData, EnrichmentItem

__all__ = [
    "EnrichmentData",
    "EnrichmentItem",
    "EnrichmentProvider",
    "EnrichmentResult",
    "get_provider",
    "register_provider",
]


@dataclass
class EnrichmentResult:
    """Summary of an enrichment run.

    Attributes:
        orders_found: Number of orders/records found in the source.
        orders_matched: Number of orders successfully matched to transactions.
        orders_unmatched: Number of orders that could not be matched.
        cache_files_written: Number of enrichment cache files written.
        unmatched_details: Human-readable descriptions of unmatched orders
            for user review.
        warnings: Non-fatal issues encountered during enrichment.
        errors: Fatal issues encountered during enrichment.
    """

    orders_found: int = 0
    orders_matched: int = 0
    orders_unmatched: int = 0
    cache_files_written: int = 0
    unmatched_details: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@runtime_checkable
class EnrichmentProvider(Protocol):
    """Protocol that all enrichment providers must satisfy.

    Providers scrape an external data source for a given month, match the
    scraped data to bank transactions, and write enrichment cache files.
    """

    @property
    def name(self) -> str:
        """Short identifier for this provider, e.g. ``"amazon"``."""
        ...

    def enrich(
        self,
        month: str,
        root: Path,
        transactions: list[dict] | None = None,
    ) -> EnrichmentResult:
        """Run enrichment for *month* and write cache files under *root*.

        Args:
            month: Target month as ``"YYYY-MM"`` string.
            root: Project root directory (enrichment-cache dir lives here).
            transactions: Optional pre-loaded transactions to match against.

        Returns:
            An :class:`EnrichmentResult` summarizing what was found and matched.
        """
        ...


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type] = {}


def register_provider(name: str, cls: type) -> None:
    """Register an enrichment provider class under *name*."""
    _PROVIDERS[name] = cls


def get_provider(name: str) -> type:
    """Look up an enrichment provider class by *name*.

    Raises:
        KeyError: If no provider is registered under *name*.
    """
    if name not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS)) or "(none)"
        raise KeyError(
            f"Unknown enrichment provider {name!r}. Available: {available}"
        )
    return _PROVIDERS[name]


# Register built-in providers on import.
def _register_builtins() -> None:
    from expense_tracker.enrichment.amazon import AmazonEnrichmentProvider

    register_provider("amazon", AmazonEnrichmentProvider)


_register_builtins()
