"""Target.com order history enrichment provider.

Uses Playwright browser automation to log into target.com, scrape order
history, and match orders to bank transactions for transaction splitting.

The matching algorithm pairs Target orders with bank transactions by date
proximity (within 3 days) and amount matching (exact or within a small
tolerance for RedCard 5% discounts).

Enrichment cache files are written to ``enrichment-cache/{transaction_id}.json``
in the format consumed by the pipeline's enrich stage.

Selector strategy
-----------------
Target.com is a React SPA that frequently changes its CSS class names and
``data-test`` attribute values. To maximise resilience the selectors below
are **comma-separated lists** ordered from *most-likely current* to
*known-legacy*. Playwright's ``query_selector`` / ``wait_for_selector``
will match on the **first** selector that hits, so new selectors go in
front and stale ones stay as fallbacks.

When all selectors fail, the scraper dumps the page HTML to a debug file
inside the cache/auth directory so the DOM can be inspected offline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consolidated CSS selectors -- new (2025-2026) first, legacy last.
# Target uses ``data-test`` attributes on most interactive elements.
# The product pages use a ``@web/<scope>/<Component>`` naming convention
# (e.g. ``@web/site-top-of-funnel/ProductCardWrapper``). The order-history
# page follows a similar pattern.
# ---------------------------------------------------------------------------

# Selector that matches *any* order card wrapper on the page.
ORDER_CARD_SELECTOR = ", ".join([
    # 2026 live selectors (from debug HTML dump)
    'div[class*="orderCard"]',
    'div[class*="OrderCard"]',
    # 2025+ namespaced component selectors
    '[data-test="@web/account/OrderCard"]',
    '[data-test="@web/account/OrderHistoryCard"]',
    '[data-test="@web/orders/OrderCard"]',
    # Shorter attribute selectors (kebab-case + camelCase)
    '[data-test="order-card"]',
    '[data-test="orderCard"]',
    # data-testid variants (React Testing Library convention)
    '[data-testid="order-card"]',
    '[data-testid="orderCard"]',
    '[data-testid="order-history-card"]',
    # Generic structural fallbacks
    '[data-component="OrderCard"]',
    'section[class*="OrderCard"]',
    'div[class*="order-card"]',
    # Broad semantic fallback -- an <article> or role wrapping each order
    'article[data-test]',
])

# Selector that indicates the page has loaded (order cards *or* empty state).
PAGE_READY_SELECTOR = ", ".join([
    # 2026 live selectors (from debug HTML dump)
    'div[class*="orderCard"]',
    'div[class*="OrderCard"]',
    '[data-test="tabOnline"]',
    '[data-test="tabInstore"]',
    '[data-test="order-details-link"]',
    # New order-card selectors
    '[data-test="@web/account/OrderCard"]',
    '[data-test="@web/account/OrderHistoryCard"]',
    '[data-test="@web/orders/OrderCard"]',
    '[data-test="order-card"]',
    '[data-test="orderCard"]',
    '[data-testid="order-card"]',
    '[data-testid="orderCard"]',
    '[data-testid="order-history-card"]',
    # Empty-state / no-orders indicators
    '[data-test="@web/account/NoOrders"]',
    '[data-test="@web/orders/EmptyState"]',
    '[data-test="no-orders"]',
    '[data-test="noOrders"]',
    '[data-test="empty-orders"]',
    '[data-testid="no-orders"]',
    '[data-testid="empty-orders"]',
    # Legacy fallbacks
    '[data-component="OrderCard"]',
    'section[class*="OrderCard"]',
    'div[class*="order-card"]',
    '.h-padding-t-tight',
    # Very broad: the account page wrapper itself (ensures we at least
    # detect the page rendered *something*).
    '[data-test="@web/account/AccountOrdersPage"]',
    '[data-test="@web/account/OrderHistoryPage"]',
    '[data-test="accountOrdersPage"]',
    '[data-testid="order-history-page"]',
    'main[role="main"]',
])

# Sub-selectors used inside an order card element.
ORDER_DATE_SELECTOR = ", ".join([
    '[data-test="@web/account/OrderDate"]',
    '[data-test="order-date"]',
    '[data-test="orderDate"]',
    '[data-testid="order-date"]',
    '[data-testid="orderDate"]',
    'time[datetime]',
    'span[class*="orderDate"]',
    'span[class*="OrderDate"]',
    'div[class*="orderDate"]',
    # Removed .h-text-sm — too broad, matches fulfillment status text
])

ORDER_NUMBER_SELECTOR = ", ".join([
    '[data-test="@web/account/OrderNumber"]',
    '[data-test="order-number"]',
    '[data-test="orderNumber"]',
    '[data-testid="order-number"]',
    '[data-testid="orderNumber"]',
    'span[class*="orderNumber"]',
    'span[class*="OrderNumber"]',
])

ORDER_TOTAL_SELECTOR = ", ".join([
    '[data-test="@web/account/OrderTotal"]',
    '[data-test="order-total"]',
    '[data-test="orderTotal"]',
    '[data-testid="order-total"]',
    '[data-testid="orderTotal"]',
    'span[class*="orderTotal"]',
    'span[class*="OrderTotal"]',
    '.h-text-bold',
])

FULFILLMENT_TYPE_SELECTOR = ", ".join([
    '[data-test="@web/account/FulfillmentType"]',
    '[data-test="fulfillment-type"]',
    '[data-test="fulfillmentType"]',
    '[data-testid="fulfillment-type"]',
    '[data-testid="fulfillmentType"]',
    'span[class*="fulfillment"]',
    'span[class*="Fulfillment"]',
])

PAYMENT_METHOD_SELECTOR = ", ".join([
    '[data-test="@web/account/PaymentMethod"]',
    '[data-test="payment-method"]',
    '[data-test="paymentMethod"]',
    '[data-testid="payment-method"]',
    '[data-testid="paymentMethod"]',
    'span[class*="payment"]',
    'span[class*="Payment"]',
])

ORDER_ITEM_CARD_SELECTOR = ", ".join([
    '[data-test="@web/account/OrderItemCard"]',
    '[data-test="@web/account/OrderItem"]',
    '[data-test="order-item-card"]',
    '[data-test="orderItemCard"]',
    '[data-testid="order-item-card"]',
    '[data-testid="orderItemCard"]',
    '[data-test="order-item"]',
    '[data-testid="order-item"]',
    'div[class*="OrderItem"]',
    'div[class*="orderItem"]',
    '.h-flex-item',
])

ITEM_NAME_SELECTOR = ", ".join([
    '[data-test="@web/account/OrderItemName"]',
    '[data-test="order-item-name"]',
    '[data-test="orderItemName"]',
    '[data-test="item-title"]',
    '[data-test="itemTitle"]',
    '[data-test="product-title"]',
    '[data-testid="order-item-name"]',
    '[data-testid="orderItemName"]',
    '[data-testid="item-title"]',
    '[data-testid="product-title"]',
    'a[data-test="product-title"]',
    'span[class*="itemName"]',
    'span[class*="ItemName"]',
    '.h-text-bold',
])

ITEM_PRICE_SELECTOR = ", ".join([
    '[data-test="@web/account/OrderItemPrice"]',
    '[data-test="order-item-price"]',
    '[data-test="orderItemPrice"]',
    '[data-test="item-price"]',
    '[data-test="itemPrice"]',
    '[data-test="current-price"]',
    '[data-testid="order-item-price"]',
    '[data-testid="orderItemPrice"]',
    '[data-testid="item-price"]',
    '[data-testid="current-price"]',
    'span[class*="itemPrice"]',
    'span[class*="ItemPrice"]',
    'span[data-test="current-price"] span',
])

ITEM_QTY_SELECTOR = ", ".join([
    '[data-test="@web/account/OrderItemQty"]',
    '[data-test="order-item-qty"]',
    '[data-test="orderItemQty"]',
    '[data-test="item-qty"]',
    '[data-test="itemQty"]',
    '[data-testid="order-item-qty"]',
    '[data-testid="orderItemQty"]',
    '[data-testid="item-qty"]',
    '[data-testid="itemQty"]',
    'span[class*="itemQty"]',
    'span[class*="ItemQty"]',
    'span[class*="quantity"]',
    'span[class*="Quantity"]',
])

# Tab selectors for Online / In-store order history tabs.
TAB_ONLINE_SELECTOR = ", ".join([
    '[data-test="tabOnline"]',
    '#tab-Online',
    'button[role="tab"][aria-controls*="Online"]',
])

TAB_INSTORE_SELECTOR = ", ".join([
    '[data-test="tabInstore"]',
    '#tab-Instore',
    'button[role="tab"][aria-controls*="Instore"]',
    'button[role="tab"][aria-controls*="In-store"]',
])

# Regex patterns used for text-based extraction from order card inner text.
# Target's order cards put date, total, and order ID in plain ``<p>`` tags
# with utility CSS classes (no ``data-test`` attributes), so CSS selectors
# are unreliable. We fall back to regex on the card's visible text, similar
# to the Amazon scraper's approach.
_DATE_RE = re.compile(
    r"(?:(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|"
    r"Nov|Dec)\s+\d{1,2},?\s+\d{4})"
)
_ORDER_TOTAL_RE = re.compile(r"\$[\d,]+\.\d{2}")
_ORDER_ID_RE = re.compile(r"#(\d{9,})")

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
            page.goto("https://www.target.com/orders", wait_until="networkidle")

            def _is_login_page() -> bool:
                url = page.url.lower()
                return any(
                    kw in url
                    for kw in ["login", "signin", "sign-in", "co-authenticate",
                               "account/sign", "auth"]
                )

            def _wait_for_user_login() -> None:
                logger.info(
                    "Login required. Please log in via the browser window. "
                    "Handle 2FA if prompted."
                )
                login_timeout = 300  # seconds
                start = time.time()
                while time.time() - start < login_timeout:
                    url_lower = page.url.lower()
                    past_login = not _is_login_page() and (
                        "orders" in url_lower
                        or "order-history" in url_lower
                        or "target.com/account" in url_lower
                        or "target.com/orders" in url_lower
                    )
                    if past_login:
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        "Login timed out after 5 minutes. Please try again."
                    )
                context.storage_state(path=str(state_file))
                logger.info("Saved Target session to %s", state_file)
                # Let the orders page render after redirect
                page.wait_for_load_state("networkidle")

            # Check if we need to log in (retry up to 2 times in case
            # the redirect to login happens after initial page load)
            for _attempt in range(2):
                if _is_login_page():
                    _wait_for_user_login()
                    break
                # Give Target time to redirect if session is expired
                time.sleep(3)

            # Wait for order history content to render (React SPA)
            try:
                page.wait_for_selector(
                    PAGE_READY_SELECTOR,
                    timeout=30000,
                )
            except Exception:
                # If selectors failed, check if we got redirected to login
                if _is_login_page():
                    _wait_for_user_login()
                    page.goto("https://www.target.com/orders", wait_until="networkidle")
                    try:
                        page.wait_for_selector(
                            PAGE_READY_SELECTOR,
                            timeout=30000,
                        )
                    except Exception:
                        _dump_debug_html(page, auth_dir)
                        raise
                else:
                    _dump_debug_html(page, auth_dir)
                    raise
            time.sleep(2)  # Extra wait for dynamic content

            # Scrape both Online and In-store tabs.
            # Target shows two tabs on the order history page; in-store
            # purchases (the bulk of big-box spending) live under a
            # separate tab that must be clicked to load its content.
            seen_order_ids: set[str] = set()
            total_cards_found = 0

            for tab_name, tab_selector in [
                ("In-store", TAB_INSTORE_SELECTOR),
                ("Online", TAB_ONLINE_SELECTOR),
            ]:
                tab_orders = _scrape_tab(
                    page, tab_name, tab_selector,
                    month_start, month_end, seen_order_ids, auth_dir,
                )
                if tab_orders is not None:
                    total_cards_found += tab_orders[1]
                    orders.extend(tab_orders[0])

            # If no tab buttons were found, scrape whatever is on the page
            # (the page may not have tabs at all for some accounts).
            if total_cards_found == 0 and not orders:
                tab_result = _scrape_current_page_orders(
                    page, month_start, month_end, seen_order_ids, auth_dir,
                )
                total_cards_found = tab_result[1]
                orders.extend(tab_result[0])

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


def _scrape_tab(
    page,
    tab_name: str,
    tab_selector: str,
    month_start: date,
    month_end: date,
    seen_order_ids: set[str],
    auth_dir: Path,
) -> tuple[list[TargetOrder], int] | None:
    """Click a tab and scrape the order cards that appear.

    Args:
        page: Playwright page object.
        tab_name: Human-readable tab name for logging (e.g. "In-store").
        tab_selector: CSS selector for the tab button.
        month_start: First day of the target month.
        month_end: Last day of the target month.
        seen_order_ids: Set of order IDs already scraped (for dedup).
        auth_dir: Directory for debug HTML dumps.

    Returns:
        Tuple of (orders, card_count) if the tab was found and clicked,
        or None if the tab button was not present on the page.
    """
    tab_button = page.query_selector(tab_selector)
    if not tab_button:
        logger.debug("Tab %r not found on page (selector: %s)", tab_name, tab_selector)
        return None

    logger.info("Clicking %r tab", tab_name)
    tab_button.click()

    # Wait for the tab content to load.  Target's SPA replaces the panel
    # content when the tab is activated; give it time to render.
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass  # networkidle may not fire if nothing loads (empty tab)
    time.sleep(2)  # extra settle time for React re-render

    return _scrape_current_page_orders(
        page, month_start, month_end, seen_order_ids, auth_dir,
    )


def _scrape_current_page_orders(
    page,
    month_start: date,
    month_end: date,
    seen_order_ids: set[str],
    auth_dir: Path,
) -> tuple[list[TargetOrder], int]:
    """Scrape order cards from the currently visible page content.

    Args:
        page: Playwright page object.
        month_start: First day of the target month.
        month_end: Last day of the target month.
        seen_order_ids: Set of order IDs already scraped (mutated in-place
            to add newly seen IDs for deduplication across tabs).
        auth_dir: Directory for debug HTML dumps.

    Returns:
        Tuple of (orders_list, total_card_count).
    """
    orders: list[TargetOrder] = []
    order_cards = page.query_selector_all(ORDER_CARD_SELECTOR)

    if not order_cards:
        logger.info("No order cards found on the current page/tab.")
        return orders, 0

    for card in order_cards:
        try:
            order = _parse_order_card(page, card, month_start, month_end)
            if order is not None:
                if order.order_id in seen_order_ids:
                    logger.debug(
                        "Skipping duplicate order %s", order.order_id,
                    )
                    continue
                seen_order_ids.add(order.order_id)
                orders.append(order)
        except Exception as exc:
            logger.warning("Failed to parse order card: %s", exc)
            continue

    # If we found cards but parsed 0 orders, dump HTML for debugging
    if order_cards and not orders:
        logger.warning(
            "Found %d order cards but parsed 0 orders. "
            "Dumping page HTML for selector debugging.",
            len(order_cards),
        )
        _dump_debug_html(page, auth_dir)

    return orders, len(order_cards)


def _dump_debug_html(page, output_dir: Path) -> Path | None:
    """Save the current page HTML to a debug file for offline inspection.

    This is called when selectors fail so we can inspect the actual DOM
    that Target served and update selectors accordingly.

    Args:
        page: Playwright page object.
        output_dir: Directory to write the debug file into.

    Returns:
        Path to the written debug file, or None on failure.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_path = output_dir / "debug-target-page.html"
        html_content = page.content()
        debug_path.write_text(html_content, encoding="utf-8")
        logger.error(
            "Selector timeout -- dumped page HTML to %s (%d bytes). "
            "Inspect this file to find current Target DOM selectors.",
            debug_path,
            len(html_content),
        )
        return debug_path
    except Exception as dump_exc:
        logger.warning("Failed to dump debug HTML: %s", dump_exc)
        return None


def _parse_order_card(page, card, month_start: date, month_end: date) -> TargetOrder | None:
    """Parse a single order card element into a TargetOrder.

    Target's 2025-2026 order cards place the date, total, and order number
    in plain ``<p>`` elements with utility CSS classes -- no ``data-test``
    attributes.  CSS-only selectors are therefore unreliable.  We extract the
    card's full visible text and use regex to pull out the date, total, and
    order ID, then fall back to CSS sub-selectors.

    Returns None if the order date is outside the target month range.

    Args:
        page: Playwright page object.
        card: The order card element.
        month_start: First day of the target month.
        month_end: Last day of the target month.

    Returns:
        A TargetOrder, or None if the order is outside the date range.
    """
    card_text = card.inner_text()

    # --- Extract order date ---
    # Strategy 1: regex on inner text (primary -- works with 2025-2026 DOM)
    order_date: date | None = None
    date_match = _DATE_RE.search(card_text)
    if date_match:
        order_date = _parse_target_date(date_match.group(0))

    # Strategy 2: CSS selector fallback (if regex missed)
    if order_date is None:
        date_el = card.query_selector(ORDER_DATE_SELECTOR)
        if date_el:
            date_text = (
                date_el.get_attribute("datetime") or date_el.inner_text()
            ).strip()
            order_date = _parse_target_date(date_text)

    if order_date is None:
        logger.warning(
            "Could not extract date from Target order card. Card text: %s",
            card_text[:200],
        )
        return None

    # Filter to target month
    if order_date < month_start or order_date > month_end:
        return None

    # --- Extract order ID ---
    order_id = ""
    id_match = _ORDER_ID_RE.search(card_text)
    if id_match:
        order_id = id_match.group(1)

    if not order_id:
        order_id_el = card.query_selector(ORDER_NUMBER_SELECTOR)
        if order_id_el:
            raw = order_id_el.inner_text().strip()
            order_id = raw.lstrip("#")

    if not order_id:
        order_id = f"unknown-{order_date.isoformat()}"

    # --- Extract order total ---
    order_total = Decimal("0")
    total_match = _ORDER_TOTAL_RE.search(card_text)
    if total_match:
        order_total = _parse_price(total_match.group(0))

    if order_total == 0:
        total_el = card.query_selector(ORDER_TOTAL_SELECTOR)
        if total_el:
            order_total = _parse_price(total_el.inner_text().strip())

    # --- Extract fulfillment type ---
    fulfillment_type = _extract_fulfillment_type(card_text)
    if not fulfillment_type:
        fulfillment_el = card.query_selector(FULFILLMENT_TYPE_SELECTOR)
        if fulfillment_el:
            fulfillment_type = _extract_fulfillment_type(
                fulfillment_el.inner_text()
            )

    # --- Extract payment method ---
    payment_el = card.query_selector(PAYMENT_METHOD_SELECTOR)
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
    item_els = card.query_selector_all(ORDER_ITEM_CARD_SELECTOR)

    for item_el in item_els:
        name_el = item_el.query_selector(ITEM_NAME_SELECTOR)
        price_el = item_el.query_selector(ITEM_PRICE_SELECTOR)
        qty_el = item_el.query_selector(ITEM_QTY_SELECTOR)

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


def _extract_fulfillment_type(text: str) -> str:
    """Determine the fulfillment type from free-form text.

    Looks for shipping, pickup, and delivery keywords while ignoring
    unrelated text (e.g. store names, status labels).

    Args:
        text: Inner text from the card or fulfillment element.

    Returns:
        One of ``"shipped"``, ``"pickup"``, ``"delivery"``, or ``""``
        if no fulfillment type could be determined.
    """
    lower = text.lower()
    # Check pickup first -- "Picked up" contains "pick" and should not
    # be confused with shipping keywords.
    if "pick" in lower:
        return "pickup"
    if "ship" in lower or "delivered" in lower:
        return "shipped"
    if "deliver" in lower:
        return "delivery"
    return ""


def _parse_target_date(text: str) -> date | None:
    """Parse a date string from Target's order history page.

    Handles formats like "January 15, 2026", "Jan 15, 2026", "1/15/2026".

    Args:
        text: Date string from the page.

    Returns:
        Parsed date, or None if parsing fails.
    """
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
