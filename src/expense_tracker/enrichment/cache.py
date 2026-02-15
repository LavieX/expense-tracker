"""Enrichment cache management.

Reads and writes enrichment cache files in JSON format.  The pipeline's
enrich stage reads these files to split aggregate transactions into
individual line items.

Cache files live at ``enrichment-cache/{transaction_id}.json`` and contain
a structured record of the matched line items from an external source
(e.g. Amazon order history).

The cache format is designed to be compatible with the pipeline's existing
``_enrich`` stage, which expects::

    {
        "items": [
            {"merchant": "...", "description": "...", "amount": "-30.00"},
            ...
        ]
    }

This module adds metadata fields (``transaction_id``, ``source``,
``order_id``, ``matched_at``) alongside the ``items`` list.  The pipeline
ignores unknown keys, so the extended format is fully backward-compatible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentItem:
    """A single line item within an enrichment cache entry.

    Attributes:
        name: Product/item name from the source (e.g. Amazon product title).
        price: Item price as a float (positive value, before sign conversion).
        quantity: Number of units purchased.
        category_hint: Optional category suggestion from the source.
        merchant: Merchant name for the split transaction (used by pipeline).
        description: Description for the split transaction (used by pipeline).
        amount: Signed amount string for the split transaction (used by pipeline).
            Negative for expenses, positive for refunds.
    """

    name: str
    price: float
    quantity: int = 1
    category_hint: str = ""
    merchant: str = ""
    description: str = ""
    amount: str = ""


@dataclass
class EnrichmentData:
    """Complete enrichment cache entry for one transaction.

    This is the full record written to ``enrichment-cache/{transaction_id}.json``.
    The ``items`` list is consumed by the pipeline's ``_enrich`` stage.

    Attributes:
        transaction_id: The bank transaction ID this enrichment matches.
        source: Enrichment provider name, e.g. ``"amazon"``.
        order_id: Source-specific order identifier.
        matched_at: ISO timestamp of when the match was made.
        items: Line items from the source, formatted for the pipeline.
    """

    transaction_id: str
    source: str
    order_id: str = ""
    matched_at: str = ""
    items: list[EnrichmentItem] = field(default_factory=list)


def write_cache_file(
    cache_dir: Path,
    data: EnrichmentData,
) -> Path:
    """Write an enrichment cache file for a single transaction.

    Creates the cache directory if it does not exist.  Overwrites any
    existing cache file for the same transaction ID.

    Args:
        cache_dir: Path to the enrichment-cache directory.
        data: The enrichment data to write.

    Returns:
        Path to the written cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Set matched_at if not already set.
    if not data.matched_at:
        data.matched_at = datetime.now().isoformat(timespec="seconds")

    cache_file = cache_dir / f"{data.transaction_id}.json"

    # Convert to dict, serializing EnrichmentItem objects.
    payload = asdict(data)

    cache_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.debug("Wrote enrichment cache: %s", cache_file)
    return cache_file


def read_cache_file(cache_file: Path) -> EnrichmentData | None:
    """Read an enrichment cache file and return an :class:`EnrichmentData`.

    Args:
        cache_file: Path to the JSON cache file.

    Returns:
        An :class:`EnrichmentData` instance, or ``None`` if the file does
        not exist or cannot be parsed.
    """
    if not cache_file.is_file():
        return None

    try:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read enrichment cache %s: %s", cache_file, exc)
        return None

    items = [
        EnrichmentItem(
            name=item.get("name", ""),
            price=float(item.get("price", 0)),
            quantity=int(item.get("quantity", 1)),
            category_hint=item.get("category_hint", ""),
            merchant=item.get("merchant", ""),
            description=item.get("description", ""),
            amount=str(item.get("amount", "0")),
        )
        for item in raw.get("items", [])
    ]

    return EnrichmentData(
        transaction_id=raw.get("transaction_id", ""),
        source=raw.get("source", ""),
        order_id=raw.get("order_id", ""),
        matched_at=raw.get("matched_at", ""),
        items=items,
    )


def list_cache_files(cache_dir: Path) -> list[Path]:
    """List all enrichment cache files in *cache_dir*.

    Returns:
        Sorted list of ``.json`` file paths.
    """
    if not cache_dir.is_dir():
        return []
    files = sorted(cache_dir.glob("*.json"))
    return files
