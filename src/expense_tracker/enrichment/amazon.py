"""Amazon order history enrichment provider.

Uses Playwright browser automation to log into Amazon, scrape order history
for a target month, and match orders to bank transactions.  Matched orders
produce enrichment cache files that the pipeline's enrich stage consumes
to split aggregate Amazon charges into individual line items.

Design decisions:
- **Headful mode**: Amazon login requires interactive 2FA/CAPTCHA handling,
  so the browser runs in headful (visible) mode.
- **Session persistence**: Cookies and browser state are stored under
  ``.auth/amazon/`` in the project directory so the user does not have to
  log in every time.
- **Conservative matching**: Orders are matched to transactions only when
  both date proximity (within 3 days) and amount (within $0.01) agree
  unambiguously.  Unmatched orders are logged for user review.
- **Pagination**: Amazon shows ~10 orders per page; the scraper follows
  pagination links to capture all orders in the date range.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from expense_tracker.enrichment.cache import (
    EnrichmentData,
    EnrichmentItem,
    write_cache_file,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# Maximum number of days between an Amazon order date and a bank transaction
# date to consider them a potential match.
DATE_PROXIMITY_DAYS = 3

# Maximum difference between order total and transaction amount to consider
# them a match (handles tax rounding).
AMOUNT_TOLERANCE = Decimal("0.01")

# Amazon order history URL template.  {year} is replaced with the target year.
ORDER_HISTORY_URL = "https://www.amazon.com/your-orders/orders?timeFilter=year-{year}"

# Directory under the project root where browser auth state is stored.
AUTH_STATE_DIR = ".auth/amazon"


@dataclass
class AmazonLineItem:
    """A single item within an Amazon order.

    Attributes:
        name: Product name/title.
        price: Item price (positive value).
        quantity: Number of units.
    """

    name: str
    price: Decimal
    quantity: int = 1


@dataclass
class AmazonOrder:
    """A scraped Amazon order.

    Attributes:
        order_id: Amazon order ID (e.g. "111-2345678-9012345").
        order_date: Date the order was placed.
        order_total: Total charge for the order.
        items: Individual line items in the order.
        account_label: Label of the Amazon account this order was scraped
            from. Used for multi-account enrichment tracking.
    """

    order_id: str
    order_date: date
    order_total: Decimal
    items: list[AmazonLineItem] = field(default_factory=list)
    account_label: str = ""


# ---------------------------------------------------------------------------
# Matching algorithm
# ---------------------------------------------------------------------------


def match_orders_to_transactions(
    orders: list[AmazonOrder],
    transactions: list[dict],
) -> list[tuple[AmazonOrder, dict]]:
    """Match Amazon orders to bank transactions by date and amount.

    Uses conservative matching: an order matches a transaction only when:
    1. The order date is within ``DATE_PROXIMITY_DAYS`` of the transaction date.
    2. The order total matches the absolute transaction amount within
       ``AMOUNT_TOLERANCE`` (to handle tax rounding).
    3. The match is unambiguous -- if multiple transactions could match the
       same order, the order is left unmatched.

    Args:
        orders: Scraped Amazon orders.
        transactions: Bank transactions as dicts with ``transaction_id``,
            ``date`` (as :class:`date`), and ``amount`` (as :class:`Decimal`)
            keys.  Amounts are negative for expenses.

    Returns:
        List of ``(order, transaction)`` tuples for successful matches.
    """
    # Build a compatibility matrix: for each order, which transactions
    # could match, and vice versa.  A match is only accepted when it is
    # unambiguous in BOTH directions (one order -> one transaction AND
    # one transaction -> one order).

    # Step 1: compute all potential (order, transaction) pairs.
    order_candidates: dict[str, list[dict]] = {}
    txn_candidates: dict[str, list[AmazonOrder]] = {}

    for order in orders:
        order_candidates.setdefault(order.order_id, [])
        for txn in transactions:
            # Date proximity check.
            day_diff = abs((order.order_date - txn["date"]).days)
            if day_diff > DATE_PROXIMITY_DAYS:
                continue

            # Amount check: order total should match absolute transaction amount.
            txn_abs = abs(txn["amount"])
            if abs(order.order_total - txn_abs) > AMOUNT_TOLERANCE:
                continue

            order_candidates[order.order_id].append(txn)
            txn_candidates.setdefault(txn["transaction_id"], []).append(order)

    # Step 2: accept only unambiguous matches (1-to-1 in both directions).
    matches: list[tuple[AmazonOrder, dict]] = []
    matched_txn_ids: set[str] = set()
    matched_order_ids: set[str] = set()

    for order in orders:
        if order.order_id in matched_order_ids:
            continue

        candidates = order_candidates.get(order.order_id, [])

        # Filter out already-matched transactions.
        available = [t for t in candidates if t["transaction_id"] not in matched_txn_ids]

        if len(available) != 1:
            if len(available) > 1:
                logger.warning(
                    "Ambiguous match for order %s ($%s on %s): "
                    "%d candidate transactions",
                    order.order_id,
                    order.order_total,
                    order.order_date,
                    len(available),
                )
            continue

        matched_txn = available[0]

        # Check the reverse: is this transaction also unambiguous?
        reverse_candidates = txn_candidates.get(matched_txn["transaction_id"], [])
        reverse_available = [
            o for o in reverse_candidates if o.order_id not in matched_order_ids
        ]
        if len(reverse_available) != 1:
            if len(reverse_available) > 1:
                logger.warning(
                    "Ambiguous match for transaction %s ($%s on %s): "
                    "%d candidate orders",
                    matched_txn["transaction_id"],
                    matched_txn["amount"],
                    matched_txn["date"],
                    len(reverse_available),
                )
            continue

        matches.append((order, matched_txn))
        matched_txn_ids.add(matched_txn["transaction_id"])
        matched_order_ids.add(order.order_id)

    return matches


# ---------------------------------------------------------------------------
# Cache file generation
# ---------------------------------------------------------------------------


def build_enrichment_data(
    order: AmazonOrder,
    transaction_id: str,
    original_merchant: str,
    account_label: str = "",
) -> EnrichmentData:
    """Convert a matched Amazon order into an :class:`EnrichmentData` for caching.

    Each line item becomes an ``EnrichmentItem`` with a signed amount
    (negative for expenses).  The items' merchant field uses the format
    ``"AMAZON - {product name}"`` for clarity in the output CSV.

    Args:
        order: The matched Amazon order.
        transaction_id: The bank transaction ID to associate with.
        original_merchant: The original merchant name from the bank transaction.
        account_label: Optional label of the Amazon account this order came
            from (for multi-account enrichment tracking).

    Returns:
        An :class:`EnrichmentData` ready to be written to the cache.
    """
    items: list[EnrichmentItem] = []

    for li in order.items:
        # Truncate long product names for readability in CSV output.
        display_name = li.name[:80] if len(li.name) > 80 else li.name
        item_total = li.price * li.quantity
        items.append(
            EnrichmentItem(
                name=li.name,
                price=float(li.price),
                quantity=li.quantity,
                category_hint="",
                merchant=f"AMAZON - {display_name}",
                description=li.name,
                amount=str(-item_total),  # Negative for expenses.
            )
        )

    # Use the order's account_label if not explicitly provided.
    label = account_label or order.account_label

    return EnrichmentData(
        transaction_id=transaction_id,
        source="amazon",
        order_id=order.order_id,
        account_label=label,
        items=items,
    )


# ---------------------------------------------------------------------------
# Amazon scraper (Playwright-based)
# ---------------------------------------------------------------------------


def _parse_month_range(month: str) -> tuple[date, date]:
    """Parse a ``YYYY-MM`` string into (first_day, last_day) of that month."""
    year, mon = month.split("-")
    year_int, mon_int = int(year), int(mon)
    first_day = date(year_int, mon_int, 1)
    if mon_int == 12:
        last_day = date(year_int + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year_int, mon_int + 1, 1) - timedelta(days=1)
    return first_day, last_day


def _parse_price(text: str) -> Decimal:
    """Parse a price string like ``"$30.00"`` or ``"30.00"`` into a Decimal."""
    cleaned = re.sub(r"[^\d.]", "", text)
    if not cleaned:
        return Decimal("0")
    return Decimal(cleaned)


def _parse_date(text: str) -> date | None:
    """Parse an Amazon date string like ``"November 15, 2025"`` into a date.

    Returns ``None`` if the string cannot be parsed.
    """
    import calendar

    # Pattern: "Month Day, Year"
    match = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", text.strip())
    if not match:
        return None

    month_name, day_str, year_str = match.groups()
    month_names = {name.lower(): num for num, name in enumerate(calendar.month_name) if num}
    mon_num = month_names.get(month_name.lower())
    if mon_num is None:
        # Try abbreviated month names.
        month_abbr = {name.lower(): num for num, name in enumerate(calendar.month_abbr) if num}
        mon_num = month_abbr.get(month_name.lower())
    if mon_num is None:
        return None

    try:
        return date(int(year_str), mon_num, int(day_str))
    except ValueError:
        return None


class AmazonEnrichmentProvider:
    """Enrichment provider that scrapes Amazon order history.

    Uses Playwright in headful mode to allow the user to handle 2FA/CAPTCHA
    during login.  Browser state (cookies, session) is persisted under
    ``.auth/amazon/`` (single account) or ``.auth/amazon-{label}/``
    (multi-account) in the project directory.

    Usage::

        provider = AmazonEnrichmentProvider()
        result = provider.enrich("2025-11", project_root)

    For multi-account usage::

        from expense_tracker.models import AmazonAccountConfig
        accounts = [
            AmazonAccountConfig(label="primary"),
            AmazonAccountConfig(label="secondary"),
        ]
        result = provider.enrich_multi_account("2025-11", project_root, accounts)
    """

    @property
    def name(self) -> str:
        return "amazon"

    def _auth_dir_for_account(self, root: Path, label: str) -> Path:
        """Return the auth state directory for a given account label.

        Single ``"default"`` accounts use the legacy path ``.auth/amazon/``.
        Named accounts use ``.auth/amazon-{label}/``.
        """
        if label == "default":
            return root / AUTH_STATE_DIR
        return root / f".auth/amazon-{label}"

    def enrich(
        self,
        month: str,
        root: Path,
        transactions: list[dict] | None = None,
    ) -> "EnrichmentResult":
        """Run Amazon enrichment for *month* with a single default account.

        This is the backward-compatible entry point. For multi-account
        enrichment, use :meth:`enrich_multi_account`.

        Args:
            month: Target month as ``"YYYY-MM"`` string.
            root: Project root directory.
            transactions: Optional list of transaction dicts to match against.
                If not provided, transactions are loaded from the pipeline.

        Returns:
            An :class:`EnrichmentResult` summarizing what was found and matched.
        """
        from expense_tracker.models import AmazonAccountConfig

        return self.enrich_multi_account(
            month=month,
            root=root,
            amazon_accounts=[AmazonAccountConfig(label="default")],
            transactions=transactions,
        )

    def enrich_multi_account(
        self,
        month: str,
        root: Path,
        amazon_accounts: list,
        transactions: list[dict] | None = None,
    ) -> "EnrichmentResult":
        """Run Amazon enrichment across multiple accounts for *month*.

        Scrapes each configured Amazon account sequentially, merges all
        orders into a single list, then runs the matching algorithm
        against the combined transaction list.

        Args:
            month: Target month as ``"YYYY-MM"`` string.
            root: Project root directory.
            amazon_accounts: List of :class:`AmazonAccountConfig` objects
                defining the accounts to scrape.
            transactions: Optional list of transaction dicts to match against.
                If not provided, transactions are loaded from the pipeline.

        Returns:
            An :class:`EnrichmentResult` with per-account stats.
        """
        from expense_tracker.enrichment import (
            AccountEnrichmentStats,
            EnrichmentResult,
        )

        first_day, last_day = _parse_month_range(month)
        cache_dir = root / "enrichment-cache"

        # Load transactions if not provided.
        if transactions is None:
            transactions = self._load_transactions(month, root)

        # Scrape orders from each Amazon account sequentially.
        all_orders: list[AmazonOrder] = []
        account_stats: list[AccountEnrichmentStats] = []
        errors: list[str] = []

        for acct in amazon_accounts:
            label = acct.label
            auth_dir = self._auth_dir_for_account(root, label)
            auth_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Scraping Amazon orders (%s)...", label)

            try:
                orders = self._scrape_orders(
                    first_day, last_day, auth_dir, cache_dir=cache_dir,
                )
                # Tag each order with the account label.
                for order in orders:
                    order.account_label = label
                all_orders.extend(orders)
                # Record per-account found count (matched count updated below).
                account_stats.append(
                    AccountEnrichmentStats(
                        label=label,
                        orders_found=len(orders),
                        orders_matched=0,
                    )
                )
            except Exception as exc:
                logger.error("Amazon scraping failed (%s): %s", label, exc)
                errors.append(f"Amazon scraping failed ({label}): {exc}")
                account_stats.append(
                    AccountEnrichmentStats(label=label, orders_found=0, orders_matched=0)
                )

        if not all_orders and errors:
            return EnrichmentResult(
                errors=errors,
                account_stats=account_stats,
            )

        # Match merged orders to transactions.
        matched = match_orders_to_transactions(all_orders, transactions)

        # Update per-account matched counts.
        matched_by_label: dict[str, int] = {}
        for order, _ in matched:
            matched_by_label[order.account_label] = (
                matched_by_label.get(order.account_label, 0) + 1
            )
        for stat in account_stats:
            stat.orders_matched = matched_by_label.get(stat.label, 0)

        # Identify unmatched orders.
        matched_order_ids = {order.order_id for order, _ in matched}
        unmatched_orders = [o for o in all_orders if o.order_id not in matched_order_ids]

        unmatched_details = []
        for order in unmatched_orders:
            item_names = ", ".join(li.name[:40] for li in order.items[:3])
            if len(order.items) > 3:
                item_names += f" (+{len(order.items) - 3} more)"
            unmatched_details.append(
                f"Order {order.order_id} ({order.order_date}, "
                f"${order.order_total}): {item_names}"
            )

        # Write cache files for matched orders.
        files_written = 0
        for order, txn in matched:
            data = build_enrichment_data(
                order=order,
                transaction_id=txn["transaction_id"],
                original_merchant=txn.get("merchant", "AMAZON"),
            )
            write_cache_file(cache_dir, data)
            files_written += 1

        return EnrichmentResult(
            orders_found=len(all_orders),
            orders_matched=len(matched),
            orders_unmatched=len(unmatched_orders),
            cache_files_written=files_written,
            unmatched_details=unmatched_details,
            errors=errors,
            account_stats=account_stats,
        )

    def _load_transactions(self, month: str, root: Path) -> list[dict]:
        """Load transactions from the pipeline for matching.

        Runs the pipeline's parse, filter, and dedup stages to get the
        transaction list for the target month.

        Returns:
            List of dicts with ``transaction_id``, ``date``, ``amount``,
            and ``merchant`` keys.
        """
        from expense_tracker.config import load_categories, load_config, load_rules
        from expense_tracker.pipeline import run

        try:
            config = load_config(root)
            categories = load_categories(root)
            rules = load_rules(root)
        except FileNotFoundError as exc:
            logger.error("Could not load config: %s", exc)
            return []

        result = run(month, config, categories, rules, root)
        return [
            {
                "transaction_id": txn.transaction_id,
                "date": txn.date,
                "amount": txn.amount,
                "merchant": txn.merchant,
            }
            for txn in result.transactions
            if not txn.is_transfer  # Don't match transfers.
        ]

    def _scrape_orders(
        self,
        first_day: date,
        last_day: date,
        auth_dir: Path,
        cache_dir: Path | None = None,
    ) -> list[AmazonOrder]:
        """Scrape Amazon order history using Playwright.

        Opens a headful browser, loads saved session state if available,
        navigates to order history, and scrapes orders within the date range.
        Handles pagination.

        Args:
            first_day: First day of the target month.
            last_day: Last day of the target month.
            auth_dir: Directory for storing/loading browser auth state.
            cache_dir: Optional cache directory for saving debug HTML dumps
                when selectors fail to match.

        Returns:
            List of :class:`AmazonOrder` objects within the date range.
        """
        from playwright.sync_api import sync_playwright

        storage_state_file = auth_dir / "state.json"

        orders: list[AmazonOrder] = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)

            # Load saved session state if available.
            context_kwargs: dict = {}
            if storage_state_file.is_file():
                context_kwargs["storage_state"] = str(storage_state_file)
                logger.info("Loading saved Amazon session")

            context = browser.new_context(**context_kwargs)
            page = context.new_page()

            try:
                # Navigate to order history for the target year.
                year = first_day.year
                page.goto(ORDER_HISTORY_URL.format(year=year), wait_until="domcontentloaded")

                # Log in if needed (may require multiple rounds for 2FA).
                for _attempt in range(3):
                    if self._needs_login(page):
                        logger.info(
                            "Amazon login required. Please log in using the browser window."
                        )
                        self._wait_for_login(page)
                        # Save session state after successful login.
                        context.storage_state(path=str(storage_state_file))
                        logger.info("Saved Amazon session state")

                        # Navigate to order history after login.
                        page.goto(
                            ORDER_HISTORY_URL.format(year=year),
                            wait_until="domcontentloaded",
                        )
                    else:
                        break

                # Scrape orders across all pages.
                orders = self._scrape_all_pages(
                    page, first_day, last_day, cache_dir=cache_dir,
                )

                # Save updated session state.
                context.storage_state(path=str(storage_state_file))

            finally:
                context.close()
                browser.close()

        return orders

    def _needs_login(self, page: "Page") -> bool:  # noqa: F821
        """Check if the current page is an Amazon auth/challenge page."""
        url = page.url.lower()
        auth_indicators = [
            "/ap/signin", "/ap/mfa", "/ap/challenge", "/ap/cvf",
            "/ap/forgotpassword",
        ]
        return any(indicator in url for indicator in auth_indicators)

    def _wait_for_login(self, page: "Page") -> None:  # noqa: F821
        """Wait for the user to complete Amazon login (including 2FA).

        Blocks until the page URL is no longer an auth/challenge page,
        indicating the user has successfully authenticated.
        """
        logger.info("Waiting for login to complete (handle 2FA if prompted)...")

        auth_indicators = [
            "/ap/signin", "/ap/mfa", "/ap/challenge", "/ap/cvf",
            "/ap/forgotpassword",
        ]

        def _is_past_auth(url: str) -> bool:
            lower = url.lower()
            return not any(ind in lower for ind in auth_indicators)

        # Wait up to 5 minutes for the user to complete login + 2FA.
        page.wait_for_url(_is_past_auth, timeout=300_000)
        logger.info("Login completed successfully")

    def _scrape_all_pages(
        self,
        page: "Page",  # noqa: F821
        first_day: date,
        last_day: date,
        cache_dir: Path | None = None,
    ) -> list[AmazonOrder]:
        """Scrape orders from all pages of Amazon order history.

        Follows pagination links until no more pages are available or
        all orders within the date range have been found.

        Args:
            page: Playwright page positioned on the order history.
            first_day: Start of the target date range.
            last_day: End of the target date range.
            cache_dir: Optional cache directory for saving debug HTML dumps
                when selectors fail to match.

        Returns:
            List of :class:`AmazonOrder` objects within the date range.
        """
        all_orders: list[AmazonOrder] = []
        page_num = 0

        while True:
            page_num += 1
            logger.info("Scraping order history page %d", page_num)

            # Wait for order cards to load.  Amazon uses several different
            # CSS class patterns depending on the layout served; try them
            # from newest to oldest.
            try:
                page.wait_for_selector(
                    "div.order-card, div.order, "
                    ".js-order-card, "
                    "[data-component='orderCard'], "
                    "#ordersContainer, "
                    ".your-orders-content-container",
                    timeout=30_000,
                )
            except Exception:
                # Dump current page HTML to a debug file so we can inspect
                # Amazon's actual DOM on future failures.
                if cache_dir is not None:
                    debug_path = cache_dir / "debug-amazon-page.html"
                    try:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        debug_path.write_text(
                            page.content(), encoding="utf-8"
                        )
                        logger.error(
                            "Order card selector timed out. "
                            "Page HTML saved to %s for debugging.",
                            debug_path,
                        )
                    except Exception as dump_exc:
                        logger.error(
                            "Failed to save debug HTML: %s", dump_exc
                        )
                raise

            page_orders = self._scrape_page_orders(page, first_day, last_day)
            all_orders.extend(page_orders)

            # If first page found no orders, dump HTML for debugging.
            if page_num == 1 and not page_orders and cache_dir is not None:
                debug_path = cache_dir / "debug-amazon-page.html"
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    debug_path.write_text(page.content(), encoding="utf-8")
                    logger.warning(
                        "Page 1 had 0 parseable orders. "
                        "HTML saved to %s for selector debugging.",
                        debug_path,
                    )
                except Exception as dump_exc:
                    logger.warning("Failed to save debug HTML: %s", dump_exc)

            # Check for next page.  Amazon's pagination uses <ul class="a-pagination">
            # with the last <li> containing the "next" link.
            next_button = page.query_selector(
                "ul.a-pagination li.a-last a, "
                ".a-pagination .a-last a, "
                "li.a-last a"
            )
            if next_button is None:
                break

            # Check if the next button is disabled.
            parent_li = next_button.evaluate_handle(
                "el => el.closest('li')"
            ).as_element()
            if parent_li and "a-disabled" in (parent_li.get_attribute("class") or ""):
                break

            next_button.click()
            page.wait_for_load_state("domcontentloaded")

        return all_orders

    def _scrape_page_orders(
        self,
        page: "Page",  # noqa: F821
        first_day: date,
        last_day: date,
    ) -> list[AmazonOrder]:
        """Scrape orders from the current page of order history.

        Args:
            page: Playwright page with order cards loaded.
            first_day: Start of the target date range.
            last_day: End of the target date range.

        Returns:
            List of :class:`AmazonOrder` objects within the date range
            found on this page.
        """
        orders: list[AmazonOrder] = []

        # Amazon order cards have various CSS class patterns.  The
        # ``div.order-card`` and ``div.order`` selectors are the primary
        # ones used in Amazon's current (2025) layout.  Older selectors
        # like ``.js-order-card`` are kept as fallbacks since Amazon may
        # serve different HTML to different users.
        order_cards = page.query_selector_all(
            "div.order-card, div.order, "
            ".js-order-card, "
            "[data-component='orderCard']"
        )

        for card in order_cards:
            try:
                order = self._parse_order_card(card)
                if order is None:
                    continue

                # Filter to target month range.
                if first_day <= order.order_date <= last_day:
                    orders.append(order)

            except Exception as exc:
                logger.warning("Failed to parse order card: %s", exc)

        return orders

    def _parse_order_card(self, card) -> AmazonOrder | None:
        """Parse a single order card element into an :class:`AmazonOrder`.

        Returns ``None`` if essential fields cannot be extracted.

        Strategy: Amazon's order card HTML uses generic CSS classes
        (e.g. ``a-color-secondary``) for both labels and values, making
        CSS-only selection unreliable.  We extract the card's full text
        and use regex to pull out the date, total, and order ID.  CSS
        selectors are used as targeted fallbacks.
        """
        card_text = card.inner_text()

        # --- Extract order date via regex on card text ---
        order_date = None
        date_match = re.search(
            r"(?:Order\s*placed|Ordered\s*on)[:\s]*"
            r"(\w+\s+\d{1,2},?\s+\d{4})",
            card_text, re.IGNORECASE,
        )
        if date_match:
            order_date = _parse_date(date_match.group(1))
        if order_date is None:
            # Broader fallback: any "Month Day, Year" in the card text.
            date_match = re.search(
                r"((?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2},?\s+\d{4})",
                card_text,
            )
            if date_match:
                order_date = _parse_date(date_match.group(1))
        if order_date is None:
            return None

        # --- Extract order total via regex on card text ---
        total_match = re.search(
            r"Total[:\s]*\$?([\d,]+\.\d{2})", card_text, re.IGNORECASE,
        )
        if total_match:
            order_total = _parse_price(total_match.group(0))
        else:
            # Broader fallback: first dollar amount in the card.
            price_match = re.search(r"\$[\d,]+\.\d{2}", card_text)
            order_total = _parse_price(price_match.group(0)) if price_match else Decimal("0")
        if order_total == 0:
            return None

        # --- Extract order ID ---
        order_id = ""
        # Try CSS selector first (reliable when present).
        order_id_el = card.query_selector(
            ".yohtmlc-order-id span[dir='ltr'], "
            ".yohtmlc-order-id bdi[dir='ltr'], "
            "[data-component='orderId'], "
            ".yohtmlc-order-id .value"
        )
        if order_id_el:
            order_id = order_id_el.inner_text().strip()
        if not order_id:
            # Try link href.
            order_link = card.query_selector("a[href*='orderID=']")
            if order_link:
                href = order_link.get_attribute("href") or ""
                id_match = re.search(r"orderID=([^&]+)", href)
                if id_match:
                    order_id = id_match.group(1)
        if not order_id:
            # Regex fallback on card text: Amazon order IDs are
            # 3-7-7 digit patterns like 113-4763190-6893819.
            id_match = re.search(r"\d{3}-\d{7}-\d{7}", card_text)
            if id_match:
                order_id = id_match.group(0)

        if not order_id:
            # Fallback: generate a pseudo-ID from date and total.
            order_id = f"unknown-{order_date.isoformat()}-{order_total}"

        # Extract line items.
        items = self._parse_line_items(card, order_total)

        return AmazonOrder(
            order_id=order_id,
            order_date=order_date,
            order_total=order_total,
            items=items,
        )

    def _parse_line_items(
        self, card, order_total: Decimal
    ) -> list[AmazonLineItem]:
        """Extract line items from an order card.

        If individual item prices cannot be determined, falls back to a
        single line item with the order total.
        """
        items: list[AmazonLineItem] = []

        # Look for individual item rows within the order card.  Amazon's
        # 2025 layout uses ``div.item-box`` for each line item.  Each
        # item-box contains a product image, title
        # (``.yohtmlc-product-title``), and action buttons.  Individual
        # item prices are NOT shown on the order history list page, so
        # we distribute the order total evenly across items.
        #
        # IMPORTANT: Do NOT include ``.a-fixed-left-grid-inner`` here --
        # it is a child of ``.item-box`` and would cause duplicate matches.
        item_els = card.query_selector_all(
            ".item-box, "
            "div.yohtmlc-item, "
            "[data-component='purchasedItems'] .a-fixed-left-grid, "
            "[data-testid='order-item']"
        )

        for item_el in item_els:
            # Extract item name.  Prefer the specific product-title class
            # (``.yohtmlc-product-title``) first; fall back
            # to a product-page link (``/dp/``) to avoid picking up
            # unrelated ``a-link-normal`` elements (e.g. "Buy it again").
            name_el = item_el.query_selector(
                ".yohtmlc-product-title, "
                "[data-component='itemTitle'], "
                ".yohtmlc-item a, "
                ".a-link-normal[href*='/dp/'], "
                "[data-testid='item-title']"
            )
            if name_el is None:
                continue
            name = name_el.inner_text().strip()
            if not name:
                continue

            # Extract item price.  Note: Amazon's 2025 order history page
            # does NOT display individual item prices (only the order total
            # in the header).  These selectors are kept for forward
            # compatibility in case Amazon adds per-item pricing later.
            price_el = item_el.query_selector(
                ".a-color-price, "
                ".yohtmlc-item-price, "
                "[data-component='unitPrice'] .a-text-price :not(.a-offscreen), "
                ".yohtmlc-item .a-color-price, "
                "[data-testid='item-price']"
            )
            price = Decimal("0")
            if price_el:
                price = _parse_price(price_el.inner_text())

            items.append(AmazonLineItem(name=name, price=price))

        # If no items found or no prices, create a single item.
        if not items:
            # Try to get at least the product name.
            product_el = card.query_selector(
                "[data-component='itemTitle'], "
                ".yohtmlc-item a, "
                ".yohtmlc-product-title, "
                ".a-link-normal[href*='/dp/'], "
                ".a-link-normal[href*='/gp/']"
            )
            product_name = "Amazon order"
            if product_el:
                product_name = product_el.inner_text().strip() or "Amazon order"

            items = [
                AmazonLineItem(name=product_name, price=order_total)
            ]
        elif all(item.price == 0 for item in items):
            # Prices not available; distribute order total evenly.
            per_item = order_total / len(items)
            for item in items:
                item.price = per_item

        return items
