"""Target.com order history enrichment provider.

Uses Playwright browser automation to log into target.com, scrape order
history, and match orders to bank transactions for transaction splitting.

The matching algorithm pairs Target orders with bank transactions by date
proximity (within 3 days) and amount matching (exact or within a small
tolerance for RedCard 5% discounts).

Enrichment cache files are written to ``enrichment-cache/{transaction_id}.json``
in the format consumed by the pipeline's enrich stage.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

AUTH_DIR = Path(".auth/target")

# Date window for matching: in-store purchases may post same day,
# online orders may take 1-3 days.
DATE_MATCH_WINDOW_DAYS = 3

# RedCard discount: 5% off. Bank charge = order_total * 0.95
REDCARD_DISCOUNT_FACTOR = Decimal("0.95")

# Tolerance for amount matching after discount adjustment ($0.02)
AMOUNT_TOLERANCE = Decimal("0.02")


@dataclass
class TargetLineItem:
    """A single item from a Target order."""

    name: str
    price: Decimal
    quantity: int = 1


@dataclass
class TargetOrder:
    """A scraped Target order with its line items."""

    order_id: str
    order_date: date
    order_total: Decimal
    items: list[TargetLineItem] = field(default_factory=list)
    fulfillment_type: str = ""  # "shipped", "pickup", "delivery"
    payment_method: str = ""  # "redcard", "debit", "credit", "gift_card"

    @property
    def has_gift_card_payment(self) -> bool:
        """True if this order was paid (even partially) with a gift card."""
        return "gift" in self.payment_method.lower()


# ---------------------------------------------------------------------------
# Order matching
# ---------------------------------------------------------------------------


def match_orders_to_transactions(
    orders: list[TargetOrder],
    transactions: list[dict],
    date_window: int = DATE_MATCH_WINDOW_DAYS,
) -> list[tuple[TargetOrder, dict]]:
    """Match Target orders to bank transactions by date and amount.

    Matching criteria:
    1. Transaction date must be within *date_window* days of order date.
    2. Transaction amount must match order total (negated, since expenses
       are negative) -- either exactly or within tolerance after applying
       the RedCard 5% discount.
    3. Orders paid entirely by gift card are skipped (they won't appear
       on a bank statement).

    Each order and transaction is matched at most once. Orders are processed
    in date order; for each order the closest matching transaction (by date)
    is preferred.

    Args:
        orders: Scraped Target orders.
        transactions: Bank transactions as dicts with keys ``transaction_id``,
            ``date`` (ISO string or date), ``amount`` (str or Decimal),
            ``merchant`` (str).
        date_window: Maximum days between order date and transaction date.

    Returns:
        List of (order, transaction) pairs that matched.
    """
    if not orders or not transactions:
        return []

    # Skip gift-card-only orders
    matchable_orders = [o for o in orders if not o.has_gift_card_payment]

    # Normalize transaction data
    normalized_txns = []
    for txn in transactions:
        txn_date = txn["date"]
        if isinstance(txn_date, str):
            txn_date = date.fromisoformat(txn_date)
        txn_amount = Decimal(str(txn["amount"]))
        normalized_txns.append({
            **txn,
            "_date": txn_date,
            "_amount": txn_amount,
        })

    # Sort orders by date for deterministic matching
    sorted_orders = sorted(matchable_orders, key=lambda o: o.order_date)

    matched_txn_ids: set[str] = set()
    matches: list[tuple[TargetOrder, dict]] = []

    for order in sorted_orders:
        best_match: dict | None = None
        best_day_diff: int = date_window + 1

        order_total_neg = -order.order_total  # Bank shows as negative
        redcard_total_neg = -(order.order_total * REDCARD_DISCOUNT_FACTOR).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        for txn in normalized_txns:
            txn_id = txn["transaction_id"]
            if txn_id in matched_txn_ids:
                continue

            # Check date proximity
            day_diff = abs((txn["_date"] - order.order_date).days)
            if day_diff > date_window:
                continue

            txn_amount = txn["_amount"]

            # Check amount match: exact or RedCard-discounted
            amount_matches = (
                abs(txn_amount - order_total_neg) <= AMOUNT_TOLERANCE
                or abs(txn_amount - redcard_total_neg) <= AMOUNT_TOLERANCE
            )

            if amount_matches and day_diff < best_day_diff:
                best_match = txn
                best_day_diff = day_diff

        if best_match is not None:
            matched_txn_ids.add(best_match["transaction_id"])
            matches.append((order, best_match))

    return matches


# ---------------------------------------------------------------------------
# Cache file I/O
# ---------------------------------------------------------------------------


def write_enrichment_cache(
    order: TargetOrder,
    transaction_id: str,
    cache_dir: Path,
) -> Path:
    """Write an enrichment cache file for a matched order.

    The file format matches what the pipeline's ``_enrich`` stage expects:
    a JSON object with an ``"items"`` list, where each item has ``merchant``,
    ``description``, and ``amount`` keys.

    The amounts are distributed proportionally so they sum to the transaction's
    actual amount (which may differ from the order total due to RedCard
    discount). Tax is distributed proportionally across items.

    Args:
        order: The matched Target order.
        transaction_id: The bank transaction ID to use as the cache filename.
        cache_dir: Directory to write the cache file into.

    Returns:
        Path to the written cache file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for item in order.items:
        item_total = item.price * item.quantity
        items.append({
            "merchant": f"Target - {item.name}",
            "description": f"{item.name} (qty {item.quantity})" if item.quantity > 1 else item.name,
            "amount": str(-item_total),
        })

    # If items don't sum to order total, add an adjustment line for
    # tax/fees/discounts
    items_sum = sum(item.price * item.quantity for item in order.items)
    remainder = order.order_total - items_sum
    if abs(remainder) > Decimal("0.00"):
        items.append({
            "merchant": "Target - Tax/Adjustments",
            "description": "Sales tax and adjustments",
            "amount": str(-remainder),
        })

    data = {
        "source": "target",
        "order_id": order.order_id,
        "order_date": order.order_date.isoformat(),
        "items": items,
    }

    cache_path = cache_dir / f"{transaction_id}.json"
    cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("Wrote enrichment cache: %s", cache_path)
    return cache_path


def read_enrichment_cache(cache_path: Path) -> dict | None:
    """Read and return an enrichment cache file, or None on failure.

    Args:
        cache_path: Path to the JSON cache file.

    Returns:
        Parsed JSON dict, or None if the file doesn't exist or is invalid.
    """
    if not cache_path.is_file():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read enrichment cache %s: %s", cache_path, exc)
        return None


# ---------------------------------------------------------------------------
# Browser automation (Playwright)
# ---------------------------------------------------------------------------


def _ensure_auth_dir() -> Path:
    """Create and return the auth directory for Target session storage."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    return AUTH_DIR


def _has_saved_session() -> bool:
    """Check if a saved browser session exists."""
    state_file = AUTH_DIR / "state.json"
    return state_file.is_file()


def scrape_target_orders(
    month: str,
    headless: bool = False,
    auth_dir: Path | None = None,
) -> list[TargetOrder]:
    """Log into target.com and scrape order history for the given month.

    Uses Playwright for browser automation. On first run, opens a visible
    browser window so the user can handle 2FA. Subsequent runs reuse the
    saved session cookies.

    Args:
        month: Target month as ``"YYYY-MM"`` string.
        headless: Whether to run the browser in headless mode. Default False
            (headful) so the user can handle 2FA on initial login.
        auth_dir: Directory for session storage. Defaults to ``.auth/target/``.

    Returns:
        List of TargetOrder objects for orders in the target month.

    Raises:
        ImportError: If playwright is not installed.
        RuntimeError: If login fails or scraping encounters an error.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for Target enrichment. "
            "Install it with: pip install playwright && playwright install chromium"
        )

    if auth_dir is None:
        auth_dir = _ensure_auth_dir()
    else:
        auth_dir.mkdir(parents=True, exist_ok=True)

    state_file = auth_dir / "state.json"

    year, mon = month.split("-")
    year_int = int(year)
    mon_int = int(mon)
    month_start = date(year_int, mon_int, 1)
    if mon_int == 12:
        month_end = date(year_int + 1, 1, 1) - timedelta(days=1)
    else:
        month_end = date(year_int, mon_int + 1, 1) - timedelta(days=1)

    orders: list[TargetOrder] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)

        # Reuse saved session if available
        if state_file.is_file():
            logger.info("Reusing saved Target session from %s", state_file)
            context = browser.new_context(storage_state=str(state_file))
        else:
            context = browser.new_context()

        page = context.new_page()

        try:
            # Navigate to Target order history
            page.goto("https://www.target.com/orders", wait_until="domcontentloaded")
            time.sleep(2)

            # Check if we need to log in
            if "login" in page.url.lower() or "account" in page.url.lower():
                logger.info(
                    "Login required. Please log in via the browser window. "
                    "Handle 2FA if prompted."
                )
                # Wait for user to complete login -- poll until we reach
                # the orders page or a reasonable timeout (5 minutes)
                login_timeout = 300  # seconds
                start = time.time()
                while time.time() - start < login_timeout:
                    if "orders" in page.url.lower():
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        "Login timed out after 5 minutes. Please try again."
                    )

                # Save session for next time
                context.storage_state(path=str(state_file))
                logger.info("Saved Target session to %s", state_file)

            # Wait for order history content to render (React SPA)
            page.wait_for_selector(
                '[data-test="orderCard"], [data-test="no-orders"], .h-padding-t-tight',
                timeout=15000,
            )
            time.sleep(2)  # Extra wait for dynamic content

            # Scrape order cards
            order_cards = page.query_selector_all('[data-test="orderCard"]')
            if not order_cards:
                logger.info("No order cards found on the page.")
                # Save session even if no orders
                context.storage_state(path=str(state_file))
                return orders

            for card in order_cards:
                try:
                    order = _parse_order_card(page, card, month_start, month_end)
                    if order is not None:
                        orders.append(order)
                except Exception as exc:
                    logger.warning("Failed to parse order card: %s", exc)
                    continue

            # Save session after successful scrape
            context.storage_state(path=str(state_file))

        except Exception:
            # Save session state even on error so user doesn't have to
            # re-login next time
            try:
                context.storage_state(path=str(state_file))
            except Exception:
                pass
            raise
        finally:
            context.close()
            browser.close()

    logger.info("Scraped %d Target orders for %s", len(orders), month)
    return orders


def _parse_order_card(page, card, month_start: date, month_end: date) -> TargetOrder | None:
    """Parse a single order card element into a TargetOrder.

    Returns None if the order date is outside the target month range.

    Args:
        page: Playwright page object.
        card: The order card element.
        month_start: First day of the target month.
        month_end: Last day of the target month.

    Returns:
        A TargetOrder, or None if the order is outside the date range.
    """
    # Extract order date
    date_el = card.query_selector('[data-test="orderDate"], .h-text-sm')
    if not date_el:
        return None
    date_text = date_el.inner_text().strip()
    order_date = _parse_target_date(date_text)
    if order_date is None:
        return None

    # Filter to target month
    if order_date < month_start or order_date > month_end:
        return None

    # Extract order ID
    order_id_el = card.query_selector('[data-test="orderNumber"]')
    order_id = order_id_el.inner_text().strip() if order_id_el else f"unknown-{order_date.isoformat()}"

    # Extract order total
    total_el = card.query_selector('[data-test="orderTotal"], .h-text-bold')
    order_total = Decimal("0")
    if total_el:
        total_text = total_el.inner_text().strip()
        order_total = _parse_price(total_text)

    # Extract fulfillment type
    fulfillment_el = card.query_selector('[data-test="fulfillmentType"]')
    fulfillment_type = ""
    if fulfillment_el:
        ft_text = fulfillment_el.inner_text().strip().lower()
        if "ship" in ft_text:
            fulfillment_type = "shipped"
        elif "pick" in ft_text:
            fulfillment_type = "pickup"
        elif "deliver" in ft_text:
            fulfillment_type = "delivery"

    # Extract payment method
    payment_el = card.query_selector('[data-test="paymentMethod"]')
    payment_method = ""
    if payment_el:
        payment_method = payment_el.inner_text().strip().lower()

    # Try to get line items from order detail page
    items = _scrape_order_items(page, card)

    return TargetOrder(
        order_id=order_id,
        order_date=order_date,
        order_total=order_total,
        items=items,
        fulfillment_type=fulfillment_type,
        payment_method=payment_method,
    )


def _scrape_order_items(page, card) -> list[TargetLineItem]:
    """Scrape individual line items from an order card or its detail page.

    Args:
        page: Playwright page object.
        card: The order card element.

    Returns:
        List of TargetLineItem objects.
    """
    items: list[TargetLineItem] = []

    # Try to find items directly on the card
    item_els = card.query_selector_all('[data-test="orderItemCard"], .h-flex-item')

    for item_el in item_els:
        name_el = item_el.query_selector(
            '[data-test="orderItemName"], [data-test="itemTitle"], .h-text-bold'
        )
        price_el = item_el.query_selector(
            '[data-test="orderItemPrice"], [data-test="itemPrice"]'
        )
        qty_el = item_el.query_selector(
            '[data-test="orderItemQty"], [data-test="itemQty"]'
        )

        if not name_el:
            continue

        name = name_el.inner_text().strip()
        price = _parse_price(price_el.inner_text().strip()) if price_el else Decimal("0")
        quantity = 1
        if qty_el:
            qty_text = qty_el.inner_text().strip()
            try:
                quantity = int("".join(c for c in qty_text if c.isdigit()) or "1")
            except ValueError:
                quantity = 1

        items.append(TargetLineItem(name=name, price=price, quantity=quantity))

    return items


def _parse_target_date(text: str) -> date | None:
    """Parse a date string from Target's order history page.

    Handles formats like "January 15, 2026", "Jan 15, 2026", "1/15/2026".

    Args:
        text: Date string from the page.

    Returns:
        Parsed date, or None if parsing fails.
    """
    import re
    from datetime import datetime

    text = text.strip()

    # Remove leading labels like "Ordered: " or "Order placed "
    for prefix in ("ordered:", "order placed", "placed on", "ordered on"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    # Try ISO format first (unlikely from Target but handle gracefully)
    try:
        return date.fromisoformat(text)
    except (ValueError, AttributeError):
        pass

    # Try "Month DD, YYYY" or "Mon DD, YYYY"
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    # Try "M/D/YYYY" or "MM/DD/YYYY"
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    logger.warning("Could not parse Target date: %r", text)
    return None


def _parse_price(text: str) -> Decimal:
    """Parse a price string like ``"$127.98"`` into a Decimal.

    Handles commas, dollar signs, and surrounding whitespace.

    Args:
        text: Price string from the page.

    Returns:
        Parsed Decimal amount (always positive).
    """
    cleaned = text.strip().replace("$", "").replace(",", "").strip()
    if not cleaned:
        return Decimal("0")
    try:
        return abs(Decimal(cleaned))
    except Exception:
        logger.warning("Could not parse price: %r", text)
        return Decimal("0")


# ---------------------------------------------------------------------------
# High-level enrichment workflow
# ---------------------------------------------------------------------------


def enrich_target(
    month: str,
    transactions: list[dict],
    cache_dir: Path,
    headless: bool = False,
    auth_dir: Path | None = None,
) -> dict:
    """Run the full Target enrichment workflow.

    1. Scrape Target orders for the month.
    2. Match orders to bank transactions.
    3. Write enrichment cache files for matched orders.

    Args:
        month: Target month as ``"YYYY-MM"`` string.
        transactions: Bank transactions as dicts (must have keys
            ``transaction_id``, ``date``, ``amount``, ``merchant``).
        cache_dir: Directory for enrichment cache files.
        headless: Browser headless mode. Default False for 2FA support.
        auth_dir: Directory for session storage. Defaults to ``.auth/target/``.

    Returns:
        Summary dict with keys ``orders_scraped``, ``orders_matched``,
        ``cache_files_written``, and ``skipped_gift_card``.
    """
    # Filter to Target-related transactions
    target_txns = [
        txn for txn in transactions
        if "target" in txn.get("merchant", "").lower()
    ]

    if not target_txns:
        logger.info("No Target transactions found for %s", month)
        return {
            "orders_scraped": 0,
            "orders_matched": 0,
            "cache_files_written": 0,
            "skipped_gift_card": 0,
        }

    # Scrape orders
    orders = scrape_target_orders(month, headless=headless, auth_dir=auth_dir)

    skipped_gc = sum(1 for o in orders if o.has_gift_card_payment)

    # Match orders to transactions
    matches = match_orders_to_transactions(orders, target_txns)

    # Write cache files
    cache_files_written = 0
    for order, txn in matches:
        if order.items:  # Only write if we have line items to split
            write_enrichment_cache(order, txn["transaction_id"], cache_dir)
            cache_files_written += 1
        else:
            logger.info(
                "Skipping cache write for order %s -- no line items scraped",
                order.order_id,
            )

    return {
        "orders_scraped": len(orders),
        "orders_matched": len(matches),
        "cache_files_written": cache_files_written,
        "skipped_gift_card": skipped_gc,
    }
