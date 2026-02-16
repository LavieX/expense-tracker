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
#
# Target's 2025-2026 order history page nests order cards inside a
# ``styledPageLayout`` wrapper.  Each order is wrapped in a div whose ID is
# the numeric order ID, with a child ``<div class="styles_orderCard__…">``
# (CSS-modules hash suffix changes per build).
#
# The most reliable live selector is the ``data-test="order-details-link"``
# attribute on the *inner* wrapper -- every order card has exactly one, and
# it does not appear elsewhere on the page.  We use its *parent* div (the
# one with the CSS-module ``orderCard`` class) as the card boundary so that
# ``inner_text()`` captures date, total, order ID, and fulfillment info.
ORDER_CARD_SELECTOR = ", ".join([
    # 2025-2026 live selector -- parent of data-test="order-details-link"
    # Matches the CSS-modules class ``styles_orderCard__<hash>``.
    'div[class*="orderCard"]',
    'div[class*="OrderCard"]',
    # NOTE: Do NOT include ``[data-test="order-details-link"]`` or
    # ``[data-test="store-order-details-link"]`` here. Those are *children*
    # of the ``orderCard`` div, so including them causes ``query_selector_all``
    # to return both parent and child for the same order, double-counting cards.
    # Legacy / future-proof selectors
    '[data-test="@web/account/OrderCard"]',
    '[data-test="@web/account/OrderHistoryCard"]',
    '[data-test="@web/orders/OrderCard"]',
    '[data-test="order-card"]',
    '[data-test="orderCard"]',
    '[data-testid="order-card"]',
    '[data-testid="orderCard"]',
    '[data-testid="order-history-card"]',
    '[data-component="OrderCard"]',
    'section[class*="OrderCard"]',
    'div[class*="order-card"]',
    'article[data-test]',
])

# Selector that indicates the page has loaded (order cards *or* empty state).
#
# The most reliable early-render indicators on Target's 2025-2026 order
# history page are the Online/In-store tab buttons, which appear before
# the order cards themselves finish rendering.  We put tabs first because
# they render immediately; the order-card selectors follow for pages that
# skip tabs (e.g. accounts with only one tab, or future redesigns).
PAGE_READY_SELECTOR = ", ".join([
    # 2025-2026 live tab selectors (render before order cards)
    '[data-test="tabOnline"]',
    '[data-test="tabInstore"]',
    # Tab content panel (visible once the SPA mounts the order-history view)
    '[data-test^="tab-tabContent-tab-"]',
    # Order card selectors (render after tab content loads)
    'div[class*="orderCard"]',
    'div[class*="OrderCard"]',
    '[data-test="order-details-link"]',
    '[data-test="store-order-details-link"]',
    # Page-level layout wrapper (present once orders section renders)
    'div[class*="styledPageLayout"]',
    # Legacy / future-proof order-card selectors
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
#
# Target's 2025-2026 order cards place date, total, and order number in
# *plain* ``<p>`` elements with utility CSS classes (``h-text-bold``,
# ``h-text-grayDark``, etc.) -- they have **no** ``data-test`` attributes.
# CSS sub-selectors are therefore unreliable for these fields.
#
# The primary extraction strategy (implemented in ``_parse_order_card``)
# uses **regex on inner text** rather than CSS selectors.  The selectors
# below serve as *fallback only* and are ordered from most-specific to
# least-specific.

ORDER_DATE_SELECTOR = ", ".join([
    # 2025-2026: date is in the first <p> child with bold styling
    'p.h-text-bold.h-text-lg',
    # Legacy data-test selectors
    '[data-test="@web/account/OrderDate"]',
    '[data-test="order-date"]',
    '[data-test="orderDate"]',
    '[data-testid="order-date"]',
    '[data-testid="orderDate"]',
    'time[datetime]',
    'span[class*="orderDate"]',
    'span[class*="OrderDate"]',
    'div[class*="orderDate"]',
])

ORDER_NUMBER_SELECTOR = ", ".join([
    # Legacy data-test selectors (the 2025-2026 In-store cards do not have a
    # separate order-number element; Online cards show it as a ``<p>`` that
    # also carries ``h-padding-b-default`` -- but that class is shared by the
    # price paragraph on In-store cards, so we must NOT use it as a selector).
    '[data-test="@web/account/OrderNumber"]',
    '[data-test="order-number"]',
    '[data-test="orderNumber"]',
    '[data-testid="order-number"]',
    '[data-testid="orderNumber"]',
    'span[class*="orderNumber"]',
    'span[class*="OrderNumber"]',
])

ORDER_TOTAL_SELECTOR = ", ".join([
    # 2025-2026: total is the second <p> inside the card, between date and order#
    # It uses h-text-grayDark and h-text-md but NOT h-padding-b-default (that's the order#)
    # and NOT h-text-bold (that's the date).  No unique selector exists, so regex is primary.
    # Legacy data-test selectors
    '[data-test="@web/account/OrderTotal"]',
    '[data-test="order-total"]',
    '[data-test="orderTotal"]',
    '[data-testid="order-total"]',
    '[data-testid="orderTotal"]',
    'span[class*="orderTotal"]',
    'span[class*="OrderTotal"]',
])

FULFILLMENT_TYPE_SELECTOR = ", ".join([
    # 2025-2026: fulfillment status is inside a heading span
    'span.h-text-grayDarkest',
    'h2 span.h-text-grayDarkest',
    # Legacy data-test selectors
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

# Item-level selectors.
#
# Target's 2025-2026 order history page does NOT show individual item
# prices or quantities on the order list view.  Items appear only as
# thumbnail images inside ``styles_imageBox__<hash>`` divs, with the
# product name available only in the ``<img alt="...">`` attribute.
#
# To get full item details (prices, quantities), the scraper must follow
# the "View purchase" link to the order detail page.  For the list view,
# we extract item *names* from image alt text.

ORDER_ITEM_IMAGE_SELECTOR = ", ".join([
    # 2025-2026: each item thumbnail is in an imageBox div.
    # NOTE: Do NOT include ``span[class*="itemPictureContainer"]`` here --
    # it is a *child* of the imageBox div, so including it would return two
    # elements per item and cause duplicates.
    'div[class*="imageBox"]',
    'div[class*="ImageBox"]',
    # Legacy structured item cards (may return on detail pages)
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
])

# Keep legacy structured-item selectors for detail pages or future changes.
ORDER_ITEM_CARD_SELECTOR = ORDER_ITEM_IMAGE_SELECTOR

ITEM_NAME_SELECTOR = ", ".join([
    # 2025-2026: product name is ONLY in img alt attribute on list view.
    # The img element inside the image box.
    'img[alt]',
    # Legacy data-test selectors (may appear on detail pages)
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

# ---------------------------------------------------------------------------
# Order detail page selectors.
#
# The detail page shows each item with its name, price, quantity, and image.
# Target's 2025-2026 detail page uses a different layout than the list view.
# Items are displayed in shipment groups (shipped, pickup, delivery) with
# structured item cards that include price and quantity information.
# ---------------------------------------------------------------------------

# Selector for the "View purchase" / "View order details" link on an order card.
# This link navigates from the list view to the order detail page.
ORDER_DETAIL_LINK_SELECTOR = ", ".join([
    'a[href*="/orders/"]',
    '[data-test="order-details-link"]',
    '[data-test="store-order-details-link"]',
    'a[aria-label*="View purchase"]',
    'a[aria-label*="View order"]',
])

# Selector for the detail page content wrapper (used to detect page load).
DETAIL_PAGE_READY_SELECTOR = ", ".join([
    # 2025-2026: detail page layout
    'div[class*="orderDetailPage"]',
    'div[class*="OrderDetailPage"]',
    'div[class*="orderDetail"]',
    'div[class*="OrderDetail"]',
    # Order summary section (always present on detail pages)
    '[data-test="order-summary"]',
    '[data-test="orderSummary"]',
    '[data-testid="order-summary"]',
    '[data-testid="orderSummary"]',
    'div[class*="orderSummary"]',
    'div[class*="OrderSummary"]',
    # Shipment groups contain the item details
    'div[class*="shipmentGroup"]',
    'div[class*="ShipmentGroup"]',
    'div[class*="fulfillmentGroup"]',
    'div[class*="FulfillmentGroup"]',
    # Item cards on the detail page
    'div[class*="itemDetail"]',
    'div[class*="ItemDetail"]',
    'div[class*="productDetail"]',
    'div[class*="ProductDetail"]',
    # Legacy selectors
    '[data-test="@web/account/OrderDetailPage"]',
    '[data-test="order-detail-page"]',
    '[data-testid="order-detail-page"]',
])

# Selector for individual item rows on the detail page.
# These are structured differently from the list view thumbnails.
DETAIL_ITEM_SELECTOR = ", ".join([
    # 2025-2026: item rows in shipment groups
    'div[class*="itemDetail"]',
    'div[class*="ItemDetail"]',
    'div[class*="productDetail"]',
    'div[class*="ProductDetail"]',
    'div[class*="itemRow"]',
    'div[class*="ItemRow"]',
    'div[class*="orderItem"]',
    'div[class*="OrderItem"]',
    # Legacy structured item cards
    '[data-test="@web/account/OrderItemCard"]',
    '[data-test="@web/account/OrderItem"]',
    '[data-test="order-item-card"]',
    '[data-test="orderItemCard"]',
    '[data-testid="order-item-card"]',
    '[data-testid="orderItemCard"]',
    '[data-test="order-item"]',
    '[data-testid="order-item"]',
])

# Selector for the item name on the detail page.
DETAIL_ITEM_NAME_SELECTOR = ", ".join([
    # Detail page typically has structured text elements, not just img alt
    'a[data-test="product-title"]',
    '[data-test="product-title"]',
    '[data-test="item-title"]',
    '[data-test="itemTitle"]',
    '[data-testid="product-title"]',
    '[data-testid="item-title"]',
    'a[class*="itemTitle"]',
    'a[class*="ItemTitle"]',
    'a[class*="productTitle"]',
    'a[class*="ProductTitle"]',
    'span[class*="itemName"]',
    'span[class*="ItemName"]',
    'div[class*="itemName"]',
    'div[class*="ItemName"]',
    'h3 a',
    'h4 a',
    # Fallback to img alt (detail page may also use images)
    'img[alt]',
])

# Selector for item price on the detail page.
DETAIL_ITEM_PRICE_SELECTOR = ", ".join([
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
    'span[class*="price"]',
    'span[data-test="current-price"] span',
])

# Selector for item quantity on the detail page.
DETAIL_ITEM_QTY_SELECTOR = ", ".join([
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
    'div[class*="quantity"]',
    'div[class*="Quantity"]',
])

# Delay between navigating to order detail pages (seconds).
# Prevents rate-limiting by Target's servers.
DETAIL_PAGE_NAV_DELAY = 1.5

# Tab selectors for Online / In-store order history tabs.
# Target's 2025-2026 tabs use ``data-test`` attributes AND ``id`` attributes.
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

# Tab content panel selectors (used to confirm tab content has loaded).
TAB_CONTENT_ONLINE_SELECTOR = ", ".join([
    '[data-test="tab-tabContent-tab-Online"]',
    '#tabContent-tab-Online',
])

TAB_CONTENT_INSTORE_SELECTOR = ", ".join([
    '[data-test="tab-tabContent-tab-Instore"]',
    '#tabContent-tab-Instore',
])

# "Load more" / "Show more" / pagination button selectors.
# Target may use a button to reveal additional orders instead of (or in
# addition to) infinite scroll.
LOAD_MORE_SELECTOR = ", ".join([
    'button[data-test="load-more"]',
    'button[data-test="loadMore"]',
    'button[data-test="show-more"]',
    'button[data-test="showMore"]',
    'button[data-test="@web/account/LoadMoreOrders"]',
    'button[data-test="@web/orders/LoadMore"]',
    'button[data-testid="load-more"]',
    'button[data-testid="loadMore"]',
    'button[data-testid="show-more"]',
    'button[data-testid="showMore"]',
    # Text-based fallbacks (Playwright comma-separated selectors)
    'button:has-text("Load more")',
    'button:has-text("Show more")',
    'button:has-text("View more orders")',
    'button:has-text("See more orders")',
    'a:has-text("Load more")',
    'a:has-text("Show more")',
    'a:has-text("View more orders")',
    'a:has-text("See more orders")',
    # CSS-class-based fallbacks
    'button[class*="loadMore"]',
    'button[class*="LoadMore"]',
    'button[class*="showMore"]',
    'button[class*="ShowMore"]',
    'a[class*="loadMore"]',
    'a[class*="LoadMore"]',
])

# Pagination link selectors (traditional next-page navigation).
PAGINATION_NEXT_SELECTOR = ", ".join([
    'a[data-test="next-page"]',
    'a[data-test="nextPage"]',
    'a[data-testid="next-page"]',
    'button[data-test="next-page"]',
    'button[data-test="nextPage"]',
    'button[aria-label="Next page"]',
    'a[aria-label="Next page"]',
    'a[aria-label="Next"]',
    'li.a-last a',  # Amazon-style fallback
    '[class*="pagination"] a:last-child',
    'nav[aria-label*="pagination"] a:last-child',
])

# Maximum number of scroll/load-more attempts before giving up.
# Each attempt scrolls to the bottom and waits for new content.
MAX_SCROLL_ATTEMPTS = 30

# Seconds to wait after each scroll for new content to appear.
SCROLL_WAIT_SECONDS = 2.0

# Number of consecutive scroll attempts with no new cards before stopping.
SCROLL_STABLE_THRESHOLD = 3

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
# In-store order IDs use a dash-separated format (e.g. "6028-2218-0085-0622").
# These appear in card text without a "#" prefix and are not matched by _ORDER_ID_RE.
_INSTORE_ORDER_ID_RE = re.compile(r"\b(\d{4}-\d{4}-\d{4}-\d{4})\b")

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
    detail_url: str = ""  # URL to the order detail page (for price scraping)

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

            tabs_to_scrape = [
                ("In-store", TAB_INSTORE_SELECTOR),
                ("Online", TAB_ONLINE_SELECTOR),
            ]
            for tab_idx, (tab_name, tab_selector) in enumerate(tabs_to_scrape):
                # Between tabs, reload the orders page to get a clean SPA
                # state.  _scrape_current_page_orders may have navigated
                # away (to order detail pages) and back, leaving the SPA
                # in an inconsistent state where the tab buttons exist but
                # the tab panel content is stale or missing.
                if tab_idx > 0:
                    logger.debug(
                        "Reloading order history page before %r tab.", tab_name,
                    )
                    try:
                        page.goto(
                            "https://www.target.com/orders",
                            wait_until="networkidle",
                        )
                        page.wait_for_selector(PAGE_READY_SELECTOR, timeout=15000)
                        time.sleep(2)
                    except Exception as exc:
                        logger.warning(
                            "Failed to reload orders page before %r tab: %s",
                            tab_name, exc,
                        )

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

    # Snapshot order card count and first-card text *before* clicking the
    # tab so we can detect when the SPA has actually swapped tab content.
    pre_click_card_count = len(page.query_selector_all(ORDER_CARD_SELECTOR))
    pre_click_first_text = ""
    try:
        pre_cards = page.query_selector_all(ORDER_CARD_SELECTOR)
        if pre_cards:
            pre_click_first_text = pre_cards[0].inner_text()[:200]
    except Exception:
        pass

    # Check if the tab is already selected (aria-selected="true").
    # If so, clicking it again would be a no-op on some SPA implementations,
    # but we still want to ensure the content panel is loaded.
    is_already_selected = tab_button.get_attribute("aria-selected") == "true"
    if is_already_selected:
        logger.debug("Tab %r is already selected, skipping click", tab_name)
    else:
        logger.info("Clicking %r tab", tab_name)
        tab_button.click()

    # Wait for the tab content to load.  Target's SPA replaces the panel
    # content when the tab is activated; give it time to render.
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass  # networkidle may not fire if nothing loads (empty tab)

    # Wait for the tab content panel to appear.  Target's 2025-2026 tabs
    # use aria-controls pointing to ``tabContent-tab-Online`` / ``tabContent-tab-Instore``.
    tab_content_selector = (
        TAB_CONTENT_INSTORE_SELECTOR if "In-store" in tab_name or "Instore" in tab_name
        else TAB_CONTENT_ONLINE_SELECTOR
    )
    try:
        page.wait_for_selector(tab_content_selector, timeout=10000)
    except Exception:
        logger.debug("Tab content panel not found for %r, proceeding anyway", tab_name)

    time.sleep(2)  # extra settle time for React re-render

    # After tab switch, explicitly wait for order cards to appear.
    # The tab panel may render before its order cards finish loading --
    # especially on the Online tab where orders are fetched via a
    # separate API call after the panel mounts.
    #
    # Two-phase wait: first wait for any card selector to appear, then
    # verify that the tab content has *actually changed* by polling for
    # a change in card count or card text.  This prevents the scraper
    # from reading stale cards from the previous tab while the SPA is
    # still swapping content.
    try:
        page.wait_for_selector(ORDER_CARD_SELECTOR, timeout=15000)
    except Exception:
        # No order cards appeared within 15 s.  Log diagnostics so we
        # can tell whether the tab is genuinely empty or the selectors
        # missed the DOM structure.
        logger.info(
            "No order cards found on %r tab after waiting 15 s. "
            "Tab may be empty or selectors may need updating.",
            tab_name,
        )
        # Dump the tab's HTML for offline inspection.
        _dump_debug_html(page, auth_dir)

    # When switching from one tab to another, the SPA may briefly show
    # the old tab's cards before replacing them.  Poll for up to 10 s
    # to confirm the content has changed (card count differs, or -- for
    # tabs with similar card counts -- the first card's text differs).
    if not is_already_selected:
        _wait_for_tab_content_change(
            page, pre_click_card_count, pre_click_first_text, tab_name,
        )

    result = _scrape_current_page_orders(
        page, month_start, month_end, seen_order_ids, auth_dir,
    )
    logger.info(
        "Tab %r: found %d order cards, %d matched target month.",
        tab_name, result[1], len(result[0]),
    )
    return result


def _wait_for_tab_content_change(
    page,
    pre_click_card_count: int,
    pre_click_first_text: str,
    tab_name: str,
    timeout_seconds: float = 10.0,
    poll_interval: float = 0.5,
) -> None:
    """Poll until the visible order cards differ from the pre-click state.

    Target's React SPA replaces the tab panel content asynchronously after
    a tab click.  There is a brief window where ``query_selector_all``
    returns stale cards from the previous tab.  This function waits until
    the card count changes *or* the first card's inner text changes,
    indicating the new tab's content has rendered.

    If the timeout expires without a change, execution continues anyway
    (the tab may genuinely have the same number of cards, or be empty).

    Args:
        page: Playwright page object.
        pre_click_card_count: Number of order cards visible before the
            tab was clicked.
        pre_click_first_text: Inner text (first 200 chars) of the first
            order card before the tab was clicked (empty if no cards).
        tab_name: Tab name for logging.
        timeout_seconds: Maximum time to wait for a content change.
        poll_interval: Seconds between polling attempts.
    """
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current_cards = page.query_selector_all(ORDER_CARD_SELECTOR)
        current_count = len(current_cards)

        # If the count changed, the new tab's content is (at least
        # partially) loaded.
        if current_count != pre_click_card_count:
            logger.debug(
                "Tab %r: card count changed %d -> %d after tab switch.",
                tab_name, pre_click_card_count, current_count,
            )
            # Give an extra moment for remaining cards to render.
            time.sleep(1)
            return

        # If the count is the same but the card text changed, the SPA
        # swapped the content.
        if current_cards and pre_click_first_text:
            try:
                current_first_text = current_cards[0].inner_text()[:200]
                if current_first_text != pre_click_first_text:
                    logger.debug(
                        "Tab %r: first card text changed after tab switch.",
                        tab_name,
                    )
                    time.sleep(1)
                    return
            except Exception:
                pass

        time.sleep(poll_interval)

    logger.debug(
        "Tab %r: card content did not visibly change within %.1f s "
        "after tab switch (pre-click count: %d). Proceeding anyway.",
        tab_name, timeout_seconds, pre_click_card_count,
    )


def _find_scrollable_container(page) -> str | None:
    """Find the scrollable container that holds the order cards.

    Target's React SPA may render order cards inside a scrollable ``<div>``
    rather than using the main page body scroll.  When that happens,
    ``window.scrollTo(0, document.body.scrollHeight)`` will not trigger
    lazy loading because the overflow container -- not the window -- is
    the element that needs to scroll.

    This function looks for a scrollable ancestor of the first order card.
    If found it returns a CSS selector string for that element; otherwise
    it returns ``None`` (meaning the window/body scroll is fine).

    Args:
        page: Playwright page object.

    Returns:
        A CSS selector string for the scrollable container element, or
        ``None`` if the window itself is the scroll host.
    """
    # The first item of ORDER_CARD_SELECTOR (before the first comma).
    first_card_sel = ORDER_CARD_SELECTOR.split(",")[0].strip()

    js = """
    ((cardSelector) => {
        const card = document.querySelector(cardSelector);
        if (!card) return null;
        let el = card.parentElement;
        while (el && el !== document.body && el !== document.documentElement) {
            const style = window.getComputedStyle(el);
            const overflow = style.overflowY;
            if ((overflow === 'auto' || overflow === 'scroll')
                && el.scrollHeight > el.clientHeight + 50) {
                if (el.id) return '#' + CSS.escape(el.id);
                const dt = el.getAttribute('data-test');
                if (dt) return '[data-test="' + dt + '"]';
                // No id or data-test: stamp a custom attribute so we can
                // target this element reliably on subsequent calls.
                el.setAttribute('data-expense-scroll', 'true');
                return '[data-expense-scroll="true"]';
            }
            el = el.parentElement;
        }
        return null;
    })
    """
    try:
        selector = page.evaluate(js, first_card_sel)
        if selector:
            logger.debug("Found scrollable container: %s", selector)
            return selector
    except Exception as exc:
        logger.debug("Failed to detect scrollable container: %s", exc)
    return None


def _scroll_to_bottom(page, container_selector: str | None) -> None:
    """Scroll the appropriate element to the bottom of its content.

    When a scrollable container is detected, this scrolls *both* the
    container and the window.  Some lazy-load / infinite-scroll listeners
    are attached to the window (``scroll`` or ``IntersectionObserver``
    rooted at the viewport), so scrolling only the container element may
    not trigger them.  Scrolling both is harmless and covers both cases.

    Args:
        page: Playwright page object.
        container_selector: CSS selector for a scrollable container, or
            ``None`` to scroll the window/body.
    """
    if container_selector:
        page.evaluate(
            """(selector) => {
                const el = document.querySelector(selector);
                if (el) el.scrollTop = el.scrollHeight;
                window.scrollTo(0, document.body.scrollHeight);
            }""",
            container_selector,
        )
    else:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")


def _scroll_to_top(page, container_selector: str | None) -> None:
    """Scroll the appropriate element back to the top.

    Args:
        page: Playwright page object.
        container_selector: CSS selector for a scrollable container, or
            ``None`` to scroll the window/body.
    """
    if container_selector:
        page.evaluate(
            """(selector) => {
                const el = document.querySelector(selector);
                if (el) el.scrollTop = 0;
            }""",
            container_selector,
        )
    else:
        page.evaluate("window.scrollTo(0, 0)")


def _scroll_and_load_all_orders(page, auth_dir: Path) -> None:
    """Scroll down and click "Load more" buttons to reveal all order cards.

    Target's order history page may use one or more of these patterns to
    show additional orders beyond the initial page load:

    1. **Infinite scroll** -- new order cards load as the user scrolls to
       the bottom of the page.
    2. **"Load more" / "Show more" button** -- a button at the bottom that
       fetches the next batch of orders when clicked.
    3. **Traditional pagination** -- "Next page" links (less common on
       Target but handled as a fallback).

    The function first detects whether the order list lives inside a
    scrollable ``<div>`` container (common in React SPAs) or uses the
    main window scroll.  It then scrolls the correct element.

    This function tries all three strategies in a loop until the number of
    visible order cards stabilises (no new cards after several attempts).

    Args:
        page: Playwright page object, already positioned on an order
            history tab.
        auth_dir: Directory for debug HTML dumps (unused here, reserved
            for future diagnostics).
    """
    previous_count = len(page.query_selector_all(ORDER_CARD_SELECTOR))
    stable_rounds = 0

    # Detect whether order cards live inside a scrollable container.
    scroll_container = _find_scrollable_container(page)

    logger.debug(
        "Starting scroll/load-more loop. Initial order card count: %d, "
        "scroll container: %s",
        previous_count,
        scroll_container or "window (body)",
    )

    for attempt in range(1, MAX_SCROLL_ATTEMPTS + 1):
        # --- Strategy 1: Click a "Load more" / "Show more" button ---
        load_more_clicked = False
        try:
            # Use query_selector (not wait_for_selector) -- if the button
            # is absent we fall through to scrolling immediately.
            load_more_btn = page.query_selector(LOAD_MORE_SELECTOR)
            if load_more_btn and load_more_btn.is_visible():
                logger.debug(
                    "Scroll attempt %d: clicking 'Load more' button", attempt,
                )
                load_more_btn.scroll_into_view_if_needed()
                load_more_btn.click()
                load_more_clicked = True
                # Wait for network activity triggered by the click.
                try:
                    page.wait_for_load_state("networkidle", timeout=10000)
                except Exception:
                    pass
                # Extra settle time for React to re-render the new cards.
                time.sleep(1)
        except Exception as exc:
            logger.debug("Load-more button interaction failed: %s", exc)

        # --- Strategy 2: Click a pagination "Next page" link ---
        if not load_more_clicked:
            try:
                next_link = page.query_selector(PAGINATION_NEXT_SELECTOR)
                if next_link and next_link.is_visible():
                    logger.debug(
                        "Scroll attempt %d: clicking 'Next page' link",
                        attempt,
                    )
                    next_link.scroll_into_view_if_needed()
                    next_link.click()
                    try:
                        page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                    # After pagination, wait for new cards to render.
                    try:
                        page.wait_for_selector(
                            ORDER_CARD_SELECTOR, timeout=10000,
                        )
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Pagination link interaction failed: %s", exc)

        # --- Strategy 3: Scroll to bottom (infinite scroll) ---
        # Always scroll to the bottom of the correct container, even after
        # a "Load more" click.  This ensures that any newly loaded content
        # (or a new "Load more" button below the fold) becomes visible for
        # the next iteration.  Scrolling also triggers infinite-scroll
        # listeners attached to the container or the window.
        _scroll_to_bottom(page, scroll_container)

        # Wait for any lazy-loaded content to appear.
        time.sleep(SCROLL_WAIT_SECONDS)

        # Recount order cards.
        current_count = len(page.query_selector_all(ORDER_CARD_SELECTOR))
        logger.debug(
            "Scroll attempt %d: card count %d -> %d",
            attempt, previous_count, current_count,
        )

        if current_count > previous_count:
            # New cards appeared -- reset the stability counter.
            stable_rounds = 0
            previous_count = current_count
        else:
            stable_rounds += 1
            if stable_rounds >= SCROLL_STABLE_THRESHOLD:
                logger.debug(
                    "No new cards after %d consecutive attempts; "
                    "stopping scroll loop. Final count: %d",
                    SCROLL_STABLE_THRESHOLD, current_count,
                )
                break

    # Scroll back to top so that subsequent card queries start from a
    # consistent viewport position.
    _scroll_to_top(page, scroll_container)
    time.sleep(0.5)


def _scrape_current_page_orders(
    page,
    month_start: date,
    month_end: date,
    seen_order_ids: set[str],
    auth_dir: Path,
) -> tuple[list[TargetOrder], int]:
    """Scrape order cards from the currently visible page content.

    After the initial cards render, this function scrolls and clicks
    "Load more" buttons to reveal all available order cards before
    collecting them.

    Once order cards are parsed from the list view (which only provides
    item names, not prices), the function navigates into each order's
    detail page to scrape per-item prices.  If a detail page fails to
    load or parse, the order falls back to $0-price items (the entire
    total lands in a "Tax/Adjustments" line in the cache file).

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

    # Give the SPA a moment to render order cards.  On Target's 2025-2026
    # page the order cards are inside a ``styledPageLayout`` wrapper that
    # appears after the "Buy Again" recommendations section loads.
    try:
        page.wait_for_selector(ORDER_CARD_SELECTOR, timeout=10000)
    except Exception:
        pass  # No order cards -- could be empty tab or loading delay

    # Scroll / click "Load more" to reveal all order cards before scraping.
    _scroll_and_load_all_orders(page, auth_dir)

    order_cards = page.query_selector_all(ORDER_CARD_SELECTOR)

    if not order_cards:
        logger.info("No order cards found on the current page/tab.")
        return orders, 0

    parse_failures = 0
    date_filtered = 0

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
            else:
                # _parse_order_card returns None for both date-filtered
                # and true parse failures.  Peek at card text to tell apart.
                card_text = card.inner_text()
                if _DATE_RE.search(card_text):
                    # Has a valid date -- likely just outside target month
                    date_filtered += 1
                else:
                    parse_failures += 1
        except Exception as exc:
            logger.warning("Failed to parse order card: %s", exc)
            parse_failures += 1
            continue

    if date_filtered:
        logger.debug(
            "Skipped %d order cards outside target month (%s to %s).",
            date_filtered, month_start, month_end,
        )

    # Only dump debug HTML when cards genuinely failed to parse (not just
    # filtered out by date range).
    if parse_failures > 0:
        logger.warning(
            "Found %d order cards but %d failed to parse. "
            "Dumping page HTML for selector debugging.",
            len(order_cards), parse_failures,
        )
        _dump_debug_html(page, auth_dir)

    # --- Navigate into each order's detail page to get per-item prices ---
    # The list view only shows item thumbnails (names from img alt) with no
    # prices.  The detail page has structured item cards with prices.
    orders_needing_prices = [
        o for o in orders
        if o.detail_url and any(item.price == Decimal("0") for item in o.items)
    ]

    if orders_needing_prices:
        logger.info(
            "Navigating to %d order detail page(s) to scrape item prices.",
            len(orders_needing_prices),
        )
        # Remember the list page URL so we can return to it after each
        # detail page visit.
        list_page_url = page.url

        for order in orders_needing_prices:
            _scrape_detail_page_prices(page, order, auth_dir)
            # Navigate back to the order list page before the next order.
            try:
                page.goto(list_page_url, wait_until="networkidle")
                time.sleep(1)
            except Exception as exc:
                logger.warning(
                    "Failed to navigate back to order list after order %s: %s",
                    order.order_id, exc,
                )
                # Try one more time with a fresh goto.
                try:
                    page.goto(
                        "https://www.target.com/orders",
                        wait_until="networkidle",
                    )
                    time.sleep(2)
                except Exception:
                    pass

    return orders, len(order_cards)


def _scrape_detail_page_prices(
    page,
    order: TargetOrder,
    auth_dir: Path,
) -> None:
    """Navigate to an order's detail page and scrape per-item prices.

    Updates ``order.items`` in place.  If the detail page cannot be loaded
    or parsed, the order's items are left unchanged (with $0 prices), so
    the downstream cache writer will put the entire total into a
    "Tax/Adjustments" line -- the same behaviour as before detail-page
    scraping was added.

    Args:
        page: Playwright page object.
        order: The order whose items need prices.  ``order.detail_url``
            must be set.
        auth_dir: Directory for debug HTML dumps on failure.
    """
    if not order.detail_url:
        return

    logger.info(
        "Scraping detail page for order %s: %s",
        order.order_id, order.detail_url,
    )

    # Rate-limit: pause before navigating to avoid triggering Target's
    # bot detection.
    time.sleep(DETAIL_PAGE_NAV_DELAY)

    try:
        page.goto(order.detail_url, wait_until="networkidle")
    except Exception as exc:
        logger.warning(
            "Failed to navigate to detail page for order %s: %s",
            order.order_id, exc,
        )
        return

    # Wait for the detail page to render.  Target's React SPA takes a
    # moment to hydrate the detail view; we use multiple signals to
    # detect readiness.
    detail_page_loaded = False
    try:
        page.wait_for_selector(DETAIL_PAGE_READY_SELECTOR, timeout=15000)
        detail_page_loaded = True
    except Exception:
        # The detail page may use an unexpected layout.  Try waiting for
        # any price-like text to appear on the page as a fallback signal.
        logger.debug(
            "Detail page ready selector not found for order %s; "
            "trying text-based fallback.",
            order.order_id,
        )

    # If the primary selector didn't fire, wait for any element that
    # contains a dollar-amount (the detail page always shows prices).
    if not detail_page_loaded:
        try:
            page.wait_for_function(
                """() => {
                    const text = document.body ? document.body.innerText : '';
                    return /\\$\\d+\\.\\d{2}/.test(text);
                }""",
                timeout=10000,
            )
        except Exception:
            logger.debug(
                "No price text detected on detail page for order %s after 10 s.",
                order.order_id,
            )

    # Extra settle time for React re-render of item cards.
    time.sleep(2)

    # --- Strategy 1: Structured item elements on the detail page ---
    detail_items = _extract_detail_page_items(page, order, auth_dir)

    if detail_items:
        # Successfully scraped items with prices from the detail page.
        # Replace the order's list-view items (which have $0 prices) with
        # the detail-page items that include real prices.
        logger.info(
            "Order %s: scraped %d item(s) with prices from detail page "
            "(CSS selector strategy).",
            order.order_id, len(detail_items),
        )
        order.items = detail_items
        return

    # --- Strategy 2: JavaScript DOM walk ---
    # CSS selectors are brittle with Target's frequent DOM changes.
    # This strategy uses JavaScript to walk the page DOM and find
    # elements that look like item cards (contain both a product name
    # and a dollar price).
    js_items = _extract_detail_page_items_via_js(page, order)
    if js_items:
        logger.info(
            "Order %s: scraped %d item(s) via JS DOM walk from detail page.",
            order.order_id, len(js_items),
        )
        order.items = js_items
        return

    # --- Strategy 3: Regex on full page text ---
    # If structured selectors and JS didn't find items, try parsing the
    # visible page text for item-name / price pairs.
    text_items = _extract_detail_page_items_from_text(page, order)
    if text_items:
        logger.info(
            "Order %s: scraped %d item(s) via text parsing from detail page.",
            order.order_id, len(text_items),
        )
        order.items = text_items
        return

    logger.warning(
        "Order %s: could not scrape item prices from detail page. "
        "Falling back to $0-price items from list view.",
        order.order_id,
    )
    _dump_debug_html(page, auth_dir)


def _extract_detail_page_items(
    page,
    order: TargetOrder,
    auth_dir: Path,
) -> list[TargetLineItem]:
    """Extract item details from the order detail page using CSS selectors.

    Looks for structured item elements (divs with item name, price, and
    quantity sub-elements) on the detail page.

    Args:
        page: Playwright page object, on the detail page.
        order: The order being scraped (for logging context).
        auth_dir: Directory for debug HTML dumps.

    Returns:
        List of TargetLineItem with prices populated, or an empty list
        if structured items could not be found.
    """
    items: list[TargetLineItem] = []

    item_elements = page.query_selector_all(DETAIL_ITEM_SELECTOR)
    if not item_elements:
        logger.debug(
            "No detail item elements found for order %s (selector: %s).",
            order.order_id, DETAIL_ITEM_SELECTOR[:80],
        )
        return items

    for item_el in item_elements:
        # --- Extract item name ---
        name = ""
        name_el = item_el.query_selector(DETAIL_ITEM_NAME_SELECTOR)
        if name_el:
            tag = name_el.evaluate("el => el.tagName.toLowerCase()")
            if tag == "img":
                name = (name_el.get_attribute("alt") or "").strip()
            else:
                name = name_el.inner_text().strip()

        if not name:
            # Try getting name from any img alt inside the item element.
            img_el = item_el.query_selector("img[alt]")
            if img_el:
                name = (img_el.get_attribute("alt") or "").strip()

        if not name:
            logger.debug(
                "Skipping item element with no extractable name "
                "in order %s.", order.order_id,
            )
            continue

        # Strip quantity suffix from name (e.g. " - quantity: 2").
        name, alt_qty = _parse_quantity_from_name(name)

        # --- Extract item price ---
        price = Decimal("0")
        price_el = item_el.query_selector(DETAIL_ITEM_PRICE_SELECTOR)
        if price_el:
            price = _parse_price(price_el.inner_text().strip())

        # If the CSS selector missed the price, try regex on the item
        # element's inner text for a dollar amount.
        if price == Decimal("0"):
            item_text = item_el.inner_text()
            price_matches = _ORDER_TOTAL_RE.findall(item_text)
            if price_matches:
                # Take the first price-like value (usually the item price;
                # later values might be strikethrough/original prices).
                price = _parse_price(price_matches[0])

        # --- Extract quantity ---
        quantity = 1
        qty_el = item_el.query_selector(DETAIL_ITEM_QTY_SELECTOR)
        if qty_el:
            qty_text = qty_el.inner_text().strip()
            try:
                quantity = int(
                    "".join(c for c in qty_text if c.isdigit()) or "1"
                )
            except ValueError:
                quantity = 1

        # Use the alt-text quantity if the CSS selector didn't find one.
        if alt_qty > 1 and quantity == 1:
            quantity = alt_qty

        # Also try regex on the item element's text for "Qty: N",
        # "Quantity: N", or "x N" / "xN" patterns.
        if quantity == 1:
            item_text = item_el.inner_text()
            qty_match = re.search(
                r"(?:qty|quantity)\s*[:=]?\s*(\d+)", item_text, re.IGNORECASE,
            )
            if not qty_match:
                # Try "x2", "x 2", "× 2" patterns (multiplication sign).
                qty_match = re.search(
                    r"(?:^|\s)[x\u00d7]\s*(\d+)(?:\s|$)", item_text,
                )
            if qty_match:
                try:
                    parsed_qty = int(qty_match.group(1))
                    if parsed_qty > 1:
                        quantity = parsed_qty
                except ValueError:
                    pass

        items.append(TargetLineItem(name=name, price=price, quantity=quantity))

    # Sanity check: if we found items but none have a price, discard them
    # (the selectors matched wrong elements).
    if items and all(item.price == Decimal("0") for item in items):
        logger.debug(
            "Order %s: all %d detail items have $0 price; "
            "discarding (likely wrong selectors).",
            order.order_id, len(items),
        )
        return []

    return items


def _extract_detail_page_items_via_js(
    page,
    order: TargetOrder,
) -> list[TargetLineItem]:
    """Extract item details from the detail page using a JavaScript DOM walk.

    This is a middle-ground strategy between CSS selectors (Strategy 1) and
    raw text parsing (Strategy 3).  It runs JavaScript in the browser to
    walk the DOM tree and find container elements that hold both a product
    name and a dollar price.

    The heuristic:
    1. Find all ``<img>`` elements with meaningful ``alt`` text (likely
       product images).
    2. For each image, walk up the DOM to find an ancestor that also
       contains a dollar price (``$X.XX``) in its text content.
    3. Extract the price from that ancestor, and the name from the img alt.
    4. Also look for quantity indicators (``Qty: N``) in the ancestor text.

    This is resilient to CSS class name changes because it relies on
    structural patterns (images near prices) rather than specific class
    names or data-test attributes.

    Args:
        page: Playwright page object, on the detail page.
        order: The order being scraped (for context / validation).

    Returns:
        List of TargetLineItem with prices, or empty list on failure.
    """
    js_code = """
    (() => {
        const results = [];
        const priceRe = /\\$(\\d[\\d,]*\\.\\d{2})/;
        const qtyRe = /(?:qty|quantity)\\s*[:=]?\\s*(\\d+)/i;
        // Collect all images that look like product images (have alt text
        // with at least 4 chars, not UI icons).
        const imgs = Array.from(document.querySelectorAll('img[alt]'));
        const seen = new Set();

        for (const img of imgs) {
            const alt = (img.alt || '').trim();
            // Skip short alt text (icons, logos) and duplicates.
            if (alt.length < 4 || seen.has(alt.toLowerCase())) continue;

            // Walk up the DOM to find an ancestor that contains a price.
            // Stop after 8 levels to avoid going too far up.
            let el = img.parentElement;
            let found = false;
            for (let depth = 0; el && depth < 8; depth++, el = el.parentElement) {
                if (el === document.body) break;
                const text = el.innerText || '';
                const pm = priceRe.exec(text);
                if (!pm) continue;

                // This ancestor contains a price.  Check if this looks
                // like an item container (not a page-level summary) by
                // verifying it doesn't contain too many price matches
                // (a summary section would have subtotal, tax, total, etc.).
                const allPrices = text.match(/\\$\\d[\\d,]*\\.\\d{2}/g) || [];
                if (allPrices.length > 4) continue;

                // Extract quantity.
                const qm = qtyRe.exec(text);
                const qty = qm ? parseInt(qm[1], 10) : 1;

                // Extract the price.  If there are multiple prices in this
                // container, prefer the one closest to the image (usually
                // the first one that isn't part of the item name).
                const price = pm[1].replace(/,/g, '');

                results.push({
                    name: alt,
                    price: price,
                    quantity: qty || 1
                });
                seen.add(alt.toLowerCase());
                found = true;
                break;
            }
        }

        // If image-based extraction found nothing, try a second pass:
        // look for link elements (product titles are often <a> tags) near
        // price elements.
        if (results.length === 0) {
            const links = Array.from(document.querySelectorAll('a'));
            for (const link of links) {
                const name = (link.innerText || '').trim();
                if (name.length < 4 || seen.has(name.toLowerCase())) continue;
                // Skip links that look like navigation, not products.
                if (/^(sign|log|cart|home|back|view|track|cancel)/i.test(name)) continue;
                if (/^(order|shipping|payment|return|help)/i.test(name)) continue;

                let el = link.parentElement;
                for (let depth = 0; el && depth < 6; depth++, el = el.parentElement) {
                    if (el === document.body) break;
                    const text = el.innerText || '';
                    const pm = priceRe.exec(text);
                    if (!pm) continue;
                    const allPrices = text.match(/\\$\\d[\\d,]*\\.\\d{2}/g) || [];
                    if (allPrices.length > 4) continue;

                    const qm = qtyRe.exec(text);
                    const qty = qm ? parseInt(qm[1], 10) : 1;
                    const price = pm[1].replace(/,/g, '');

                    results.push({
                        name: name,
                        price: price,
                        quantity: qty || 1
                    });
                    seen.add(name.toLowerCase());
                    break;
                }
            }
        }

        return results;
    })()
    """

    try:
        raw_items = page.evaluate(js_code)
    except Exception as exc:
        logger.debug(
            "JS DOM walk failed for order %s: %s", order.order_id, exc,
        )
        return []

    if not raw_items:
        return []

    items: list[TargetLineItem] = []
    known_list_names = {item.name.lower() for item in order.items}

    for raw in raw_items:
        name = raw.get("name", "").strip()
        if not name:
            continue

        # Strip quantity suffix from name (e.g. " - quantity: 2").
        name, alt_qty = _parse_quantity_from_name(name)

        try:
            price = Decimal(raw.get("price", "0"))
        except Exception:
            price = Decimal("0")

        quantity = raw.get("quantity", 1) or 1
        if alt_qty > 1 and quantity == 1:
            quantity = alt_qty

        if price > Decimal("0"):
            items.append(TargetLineItem(name=name, price=price, quantity=quantity))

    # Sanity check: items total should not wildly exceed the order total.
    if items:
        items_total = sum(item.price * item.quantity for item in items)
        if items_total > order.order_total * Decimal("2.0"):
            logger.debug(
                "Order %s: JS-extracted items total $%s exceeds order total "
                "$%s by >100%%; discarding as likely false positives.",
                order.order_id, items_total, order.order_total,
            )
            return []

    # Also discard if no extracted item matches any name from the list view.
    # This helps avoid picking up recommended/related product items that
    # appear on the detail page but aren't part of the order.
    if items and known_list_names:
        matched_any = False
        for item in items:
            item_lower = item.name.lower()
            for known in known_list_names:
                if len(known) < 4:
                    continue
                if known in item_lower or item_lower in known:
                    matched_any = True
                    break
            if matched_any:
                break
        if not matched_any:
            logger.debug(
                "Order %s: JS-extracted items don't match any list-view "
                "item names; discarding (likely wrong elements).",
                order.order_id,
            )
            return []

    return items


def _extract_detail_page_items_from_text(
    page,
    order: TargetOrder,
) -> list[TargetLineItem]:
    """Fallback: extract item names and prices from the detail page text.

    When CSS selectors fail (Target may have restructured the DOM), this
    function scans the full page text for patterns that look like item
    lines with associated prices.

    The heuristic looks for lines that contain a dollar amount and are
    preceded or followed by a line that looks like a product name (no
    dollar sign, non-empty, not a section header).

    Args:
        page: Playwright page object, on the detail page.
        order: The order being scraped (for context / matching).

    Returns:
        List of TargetLineItem, or empty list if parsing fails.
    """
    try:
        page_text = page.inner_text("body")
    except Exception:
        return []

    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]

    # Build a list of (name, price) pairs by scanning for price lines
    # and associating them with adjacent non-price lines.
    items: list[TargetLineItem] = []
    known_list_names = {item.name.lower() for item in order.items}
    used_name_indices: set[int] = set()

    # Lines that are clearly section headers / summary labels, not products.
    _HEADER_PREFIXES = (
        "Order", "Shipping", "Tax", "Subtotal", "Total", "Items",
        "Payment", "Delivery", "Shipped", "Picked up", "Estimated",
        "Sign", "Log", "Cart", "Help", "Return", "Track", "Cancel",
        "Back", "View all", "Contact", "Chat", "Email",
    )

    def _is_product_name_candidate(candidate: str) -> bool:
        """Return True if the line looks like it could be a product name."""
        if len(candidate) < 4:
            return False
        if _ORDER_TOTAL_RE.search(candidate):
            return False
        if not candidate[0].isalpha():
            return False
        if candidate.startswith(_HEADER_PREFIXES):
            return False
        return True

    def _is_known_name(candidate: str) -> bool:
        """Return True if the candidate matches a known item from the list view."""
        candidate_lower = candidate.lower()
        return any(
            known in candidate_lower or candidate_lower in known
            for known in known_list_names
            if len(known) >= 4
        )

    i = 0
    while i < len(lines):
        line = lines[i]
        price_match = _ORDER_TOTAL_RE.search(line)

        if price_match:
            price = _parse_price(price_match.group(0))
            name = ""
            name_idx = -1

            # Look BACKWARD first (product name before price -- most common).
            for offset in range(1, 5):
                if i - offset < 0:
                    break
                idx = i - offset
                if idx in used_name_indices:
                    continue
                candidate = lines[idx]
                if not _is_product_name_candidate(candidate):
                    continue
                if _is_known_name(candidate) or len(candidate) >= 8:
                    name = candidate
                    name_idx = idx
                    break

            # Look FORWARD if backward search failed (price before name).
            if not name:
                for offset in range(1, 5):
                    if i + offset >= len(lines):
                        break
                    idx = i + offset
                    if idx in used_name_indices:
                        continue
                    candidate = lines[idx]
                    if _ORDER_TOTAL_RE.search(candidate):
                        break  # Hit the next price line -- stop looking forward
                    if not _is_product_name_candidate(candidate):
                        continue
                    if _is_known_name(candidate) or len(candidate) >= 8:
                        name = candidate
                        name_idx = idx
                        break

            if name and price > Decimal("0"):
                name, qty = _parse_quantity_from_name(name)
                items.append(TargetLineItem(
                    name=name, price=price, quantity=qty,
                ))
                if name_idx >= 0:
                    used_name_indices.add(name_idx)
        i += 1

    # Sanity check: items total should not wildly exceed the order total.
    # If it does, the text parser likely picked up wrong prices.
    # Use a 2x threshold (not 1.5x) because detail pages may show original
    # prices before discounts were applied.
    if items:
        items_total = sum(item.price * item.quantity for item in items)
        if items_total > order.order_total * Decimal("2.0"):
            logger.debug(
                "Order %s: text-parsed items total $%s exceeds order total "
                "$%s by >100%%; discarding as likely false positives.",
                order.order_id, items_total, order.order_total,
            )
            return []

    return items


def _dump_debug_html(page, output_dir: Path) -> Path | None:
    """Save the current page HTML to a debug file for offline inspection.

    This is called when selectors fail so we can inspect the actual DOM
    that Target served and update selectors accordingly.

    Each dump gets a unique timestamp suffix so that successive dumps
    (e.g. one per tab) do not overwrite each other.

    Args:
        page: Playwright page object.
        output_dir: Directory to write the debug file into.

    Returns:
        Path to the written debug file, or None on failure.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        debug_path = output_dir / f"debug-target-page-{timestamp}.html"
        html_content = page.content()
        debug_path.write_text(html_content, encoding="utf-8")
        logger.warning(
            "Dumped page HTML to %s (%d bytes). "
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
    attributes.  CSS-only selectors are therefore unreliable.

    Extraction strategies (in priority order):

    1. **Regex on inner text** -- the card's visible text contains the date,
       total, and order ID in a predictable format.
    2. **"View purchase" link** -- the ``<a>`` element has an ``aria-label``
       like ``"View purchase made on Aug 31, 2024 for $30.52"`` and an
       ``href`` like ``"/orders/102001197478538"`` which encodes the order ID.
    3. **CSS sub-selectors** -- fallback for future DOM changes.

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

    # Grab the "View purchase" link for supplementary data extraction.
    # Its aria-label encodes date + total; its href encodes the order ID.
    # Use the broad ORDER_DETAIL_LINK_SELECTOR for resilience against DOM
    # changes, rather than a single hardcoded selector.
    view_link = card.query_selector(ORDER_DETAIL_LINK_SELECTOR)
    aria_label = ""
    link_href = ""
    if view_link:
        aria_label = view_link.get_attribute("aria-label") or ""
        link_href = view_link.get_attribute("href") or ""

    # --- Extract order date ---
    # Strategy 1: regex on inner text (primary -- works with 2025-2026 DOM)
    order_date: date | None = None
    date_match = _DATE_RE.search(card_text)
    if date_match:
        order_date = _parse_target_date(date_match.group(0))

    # Strategy 2: regex on aria-label (e.g. "View purchase made on Aug 31, 2024 for $30.52")
    if order_date is None and aria_label:
        aria_date_match = _DATE_RE.search(aria_label)
        if aria_date_match:
            order_date = _parse_target_date(aria_date_match.group(0))

    # Strategy 3: CSS selector fallback (if regex missed)
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

    # Strategy 1a: regex on inner text (matches #NNNNNNNNN -- Online orders)
    id_match = _ORDER_ID_RE.search(card_text)
    if id_match:
        order_id = id_match.group(1)

    # Strategy 1b: regex for in-store dash-format IDs (e.g. "6028-2218-0085-0622")
    if not order_id:
        instore_id_match = _INSTORE_ORDER_ID_RE.search(card_text)
        if instore_id_match:
            order_id = instore_id_match.group(1)

    # Strategy 2: extract from href
    # Online orders:  /orders/102001197478538
    # In-store orders: /orders/stores/5350-2218-0175-9554
    if not order_id and link_href:
        href_id_match = re.search(r"/orders/(?:stores/)?([\d-]+)", link_href)
        if href_id_match:
            order_id = href_id_match.group(1)

    # Strategy 3: CSS selector fallback
    if not order_id:
        order_id_el = card.query_selector(ORDER_NUMBER_SELECTOR)
        if order_id_el:
            raw = order_id_el.inner_text().strip()
            # Only use this if it looks like an order number (digits,
            # possibly with # prefix or dashes), not a price.
            cleaned = raw.lstrip("#")
            if cleaned and not cleaned.startswith("$"):
                order_id = cleaned

    if not order_id:
        order_id = f"unknown-{order_date.isoformat()}"

    # --- Extract order total ---
    order_total = Decimal("0")

    # Strategy 1: regex on inner text
    total_match = _ORDER_TOTAL_RE.search(card_text)
    if total_match:
        order_total = _parse_price(total_match.group(0))

    # Strategy 2: regex on aria-label (e.g. "...for $30.52")
    if order_total == 0 and aria_label:
        aria_total_match = _ORDER_TOTAL_RE.search(aria_label)
        if aria_total_match:
            order_total = _parse_price(aria_total_match.group(0))

    # Strategy 3: CSS selector fallback
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

    # Try to get line items from the order card
    items = _scrape_order_items(page, card)

    # Build the detail page URL for later navigation.
    # The link_href is a relative path like "/orders/102001197478538" or
    # "/orders/stores/5350-2218-0175-9554".
    detail_url = ""
    if link_href:
        if link_href.startswith("http"):
            detail_url = link_href
        elif link_href.startswith("/"):
            detail_url = f"https://www.target.com{link_href}"

    # Fallback for in-store orders: their order cards may not have a
    # clickable link, but the detail page URL is constructable from the
    # dash-format order ID (e.g. "6028-2218-0085-0622").
    if not detail_url and order_id and "-" in order_id:
        detail_url = f"https://www.target.com/orders/stores/{order_id}"

    return TargetOrder(
        order_id=order_id,
        order_date=order_date,
        order_total=order_total,
        items=items,
        fulfillment_type=fulfillment_type,
        payment_method=payment_method,
        detail_url=detail_url,
    )


def _scrape_order_items(page, card) -> list[TargetLineItem]:
    """Scrape individual line items from an order card.

    Target's 2025-2026 order list page shows items as thumbnail images
    inside ``imageBox`` divs.  The product name is available **only** as
    the ``<img alt="...">`` attribute; individual prices and quantities
    are not displayed on the list view.

    Quantities are encoded in two places:

    * The ``<img alt>`` text may contain a ``" - quantity: N"`` suffix
      (e.g. ``"Oatly Oatmilk Full Fat - 64 fl oz - quantity: 2"``).
    * The ``itemPictureContainer`` span uses a CSS custom property
      ``--quantity-content: "N"`` and a ``hasQuantity`` CSS class.

    When structured item cards with price/qty sub-elements are present
    (future redesign or order detail page), those are preferred.

    Args:
        page: Playwright page object.
        card: The order card element.

    Returns:
        List of TargetLineItem objects.  On the list view, each item will
        have ``price=0`` and ``quantity=1`` (unless quantity is encoded in
        the alt text) since price details are only available on the order
        detail page.
    """
    items: list[TargetLineItem] = []

    # Strategy 1: Try structured item cards with price/qty (legacy or detail page)
    item_els = card.query_selector_all(ORDER_ITEM_CARD_SELECTOR)

    for item_el in item_els:
        # Look for a structured name element first
        name_el = item_el.query_selector(ITEM_NAME_SELECTOR)
        price_el = item_el.query_selector(ITEM_PRICE_SELECTOR)
        qty_el = item_el.query_selector(ITEM_QTY_SELECTOR)

        if not name_el:
            continue

        # Determine name: prefer inner_text from a structured element,
        # but if the matched element is an <img>, use its alt attribute.
        tag_name = name_el.evaluate("el => el.tagName.toLowerCase()")
        if tag_name == "img":
            name = (name_el.get_attribute("alt") or "").strip()
        else:
            name = name_el.inner_text().strip()

        if not name:
            continue

        price = Decimal("0")
        if price_el:
            price = _parse_price(price_el.inner_text().strip())

        quantity = 1
        if qty_el:
            qty_text = qty_el.inner_text().strip()
            try:
                quantity = int("".join(c for c in qty_text if c.isdigit()) or "1")
            except ValueError:
                quantity = 1

        # Parse quantity from alt text suffix like " - quantity: 2"
        name, alt_qty = _parse_quantity_from_name(name)
        if alt_qty > 1 and quantity == 1:
            quantity = alt_qty

        items.append(TargetLineItem(name=name, price=price, quantity=quantity))

    # Strategy 2: If no structured items were found, try extracting names
    # from image alt text.  In-store cards do NOT have
    # ``data-test="order-images-component"``; their images live in a
    # ``packageImagesContainer`` div.  We search for both, then fall back
    # to any ``img[alt]`` inside the card.
    if not items:
        # Try order-images-component first (Online cards)
        images_container = card.query_selector('[data-test="order-images-component"]')
        # Fall back to packageImagesContainer (In-store cards)
        if not images_container:
            images_container = card.query_selector('div[class*="packageImagesContainer"]')
        # Last resort: search the entire card
        if not images_container:
            images_container = card

        img_els = images_container.query_selector_all("img[alt]")
        for img_el in img_els:
            alt = (img_el.get_attribute("alt") or "").strip()
            if alt:
                name, quantity = _parse_quantity_from_name(alt)
                items.append(TargetLineItem(
                    name=name, price=Decimal("0"), quantity=quantity,
                ))

    return items


# Regex to extract quantity suffix from item alt text.
# Matches patterns like " - quantity: 2" at the end of the string.
_QUANTITY_SUFFIX_RE = re.compile(r"\s*-\s*quantity:\s*(\d+)\s*$", re.IGNORECASE)


def _parse_quantity_from_name(name: str) -> tuple[str, int]:
    """Strip a ``" - quantity: N"`` suffix from an item name.

    Target's In-store order thumbnails encode the purchase quantity in the
    ``<img alt>`` attribute as a suffix (e.g. ``"Fresh Dekopon Mandarin
    Orange - each - quantity: 4"``).

    Args:
        name: Item name, possibly with a quantity suffix.

    Returns:
        Tuple of (cleaned_name, quantity).  If no suffix is found,
        quantity defaults to 1.
    """
    m = _QUANTITY_SUFFIX_RE.search(name)
    if m:
        qty = int(m.group(1))
        cleaned = name[:m.start()].strip()
        return cleaned, max(qty, 1)
    return name, 1


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
