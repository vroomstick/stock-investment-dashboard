"""
data/collectors/sec_edgar.py

Fetches and parses SEC EDGAR filings for all stocks in the universe.

Covers:
- Form 4  : insider buy/sell transactions
- 8-K     : material events (restatements, exec departures, investigations)
- SC 13D/G: activist investor disclosures (>5% stake)
- 10-Q/10-K: quarterly/annual reports (filing log only — financials via XBRL)

Rate limit: 10 req/sec enforced by SEC. We run at 8/sec for safety.
User-Agent header is mandatory — SEC blocks requests without it.
"""

import os
import json
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Optional


import requests
from dotenv import load_dotenv
load_dotenv()

from config.settings import SEC
from database.db import get_db, fetch_one, executemany


# ---------------------------------------------------------------------------
# Rate-limited HTTP client
# ---------------------------------------------------------------------------

_last_request_time = 0.0


def _get(url: str) -> requests.Response:
    """
    GET request with SEC rate limiting and exponential backoff on 503.

    Why exponential backoff?
    When the SEC server is overloaded it returns 503. Immediately retrying
    makes it worse. Waiting 2^attempt seconds (2s, 4s, 8s) gives the server
    time to recover. This is the industry standard retry pattern.
    """
    global _last_request_time

    # Enforce rate limit: wait if we're moving too fast
    min_interval = 1.0 / SEC["rate_limit_per_sec"]
    elapsed = time.time() - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)

    headers = {"User-Agent": SEC["user_agent"]}

    for attempt in range(SEC["retry_attempts"]):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            _last_request_time = time.time()

            if response.status_code == 200:
                return response
            elif response.status_code == 503:
                wait = SEC["retry_backoff_base"] ** attempt
                print(f"    SEC 503 — waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)
            else:
                response.raise_for_status()

        except requests.RequestException as e:
            if attempt == SEC["retry_attempts"] - 1:
                raise
            time.sleep(SEC["retry_backoff_base"] ** attempt)

    raise RuntimeError(f"Failed to fetch {url} after {SEC['retry_attempts']} attempts")


# ---------------------------------------------------------------------------
# EDGAR API calls
# ---------------------------------------------------------------------------

def _pad_cik(cik: str) -> str:
    """Zero-pad CIK to 10 digits as required by the EDGAR API."""
    return cik.strip().lstrip("0").zfill(10)


def fetch_filing_history(cik: str) -> dict:
    """
    Fetch the complete filing history for a company.
    Returns the raw JSON from the EDGAR submissions endpoint.

    The response contains a 'filings.recent' object with parallel arrays:
    accessionNumber[], filingDate[], form[], primaryDocument[], etc.
    We zip these into dicts in get_recent_filings().
    """
    url = SEC["submissions_url"].format(cik=_pad_cik(cik))
    return _get(url).json()


def get_recent_filings(cik: str, form_types: list, days_back: int = 90,
                       as_of_date: str = None) -> list[dict]:
    """
    Return filings of the specified form types filed within the last N days.
    as_of_date: anchor date for the lookback window (default: today).
                Pass this explicitly for reproducible historical reruns —
                using date.today() in historical context leaks future knowledge.
    """
    anchor = date.fromisoformat(as_of_date) if as_of_date else date.today()
    cutoff = (anchor - timedelta(days=days_back)).isoformat()

    history = fetch_filing_history(cik)
    recent = history.get("filings", {}).get("recent", {})

    if not recent:
        return []

    # EDGAR returns parallel arrays — zip into list of dicts
    keys = ["accessionNumber", "filingDate", "form", "primaryDocument", "reportDate"]
    available = [k for k in keys if k in recent]
    records = [
        dict(zip(available, values))
        for values in zip(*[recent[k] for k in available])
    ]

    return [
        r for r in records
        if r.get("form") in form_types
        and r.get("filingDate", "") >= cutoff
    ]


def fetch_xbrl_facts(cik: str) -> dict:
    """
    Fetch all XBRL-tagged financial data for a company.
    Used by the fundamental feature store to get revenue, earnings, etc.

    XBRL (eXtensible Business Reporting Language) is a standardized XML
    format that public companies use to tag their financial statements.
    The SEC requires it, which is why this data is freely available.
    """
    url = SEC["company_facts_url"].format(cik=_pad_cik(cik))
    return _get(url).json()


# ---------------------------------------------------------------------------
# Form 4 — Insider Transactions
# ---------------------------------------------------------------------------

def fetch_form4_xml(cik: str, accession_number: str, primary_doc: str) -> Optional[str]:
    """
    Fetch the raw XML of a Form 4 filing from the EDGAR archives.

    EDGAR archive URL structure:
    /Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primary_document}
    """
    cik_int = str(int(cik.lstrip("0") or "0"))
    accession_clean = accession_number.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_clean}/{primary_doc}"
    )
    try:
        return _get(url).text
    except Exception as e:
        print(f"    Could not fetch Form 4 XML: {e}")
        return None


def parse_form4(xml_content: str) -> list[dict]:
    """
    Parse a Form 4 XML document into a list of transaction dicts.

    Form 4 is an SEC filing insiders must submit within 2 business days
    of any transaction in company stock. It's one of the strongest signals
    in the system — a CEO buying $2M of stock with their own money is
    legally required to be disclosed here.

    Transaction codes we care about:
    - P: open market purchase (strong bullish signal)
    - S: open market sale (bearish signal, but noisy — could be diversification)
    - A: grant/award (not a market signal — ignore)
    """
    transactions = []

    try:
        root = ET.fromstring(xml_content)

        # Extract insider identity
        insider_name = ""
        insider_title = ""
        owner_el = root.find(".//reportingOwner")
        if owner_el is not None:
            name_el = owner_el.find(".//rptOwnerName")
            if name_el is not None:
                insider_name = (name_el.text or "").strip()

            rel_el = owner_el.find(".//reportingOwnerRelationship")
            if rel_el is not None:
                title_el = rel_el.find("officerTitle")
                if title_el is not None and title_el.text:
                    insider_title = title_el.text.strip()
                else:
                    roles = []
                    for field, label in [
                        ("isOfficer", "Officer"),
                        ("isDirector", "Director"),
                        ("isTenPercentOwner", "10% Owner"),
                    ]:
                        el = rel_el.find(field)
                        if el is not None and el.text == "1":
                            roles.append(label)
                    insider_title = ", ".join(roles)

        # Parse non-derivative transactions (actual stock, not options)
        for txn in root.findall(".//nonDerivativeTransaction"):

            def get_val(tag):
                el = txn.find(f".//{tag}")
                return el.text.strip() if el is not None and el.text else None

            txn_code = get_val("transactionCode")
            if txn_code not in ("P", "S"):
                continue  # Skip grants, awards, etc.

            try:
                shares = float(get_val("transactionShares") or 0)
                price  = float(get_val("transactionPricePerShare") or 0)
                owned  = float(get_val("sharesOwnedFollowingTransaction") or 0)
                txn_date = get_val("transactionDate")

                transactions.append({
                    "insider_name":       insider_name,
                    "insider_title":      insider_title,
                    "transaction_type":   txn_code,
                    "shares":             shares,
                    "price_per_share":    price,
                    "total_value":        shares * price,
                    "shares_owned_after": owned,
                    "transaction_date":   txn_date,
                })
            except (ValueError, TypeError):
                continue

    except ET.ParseError as e:
        print(f"    XML parse error on Form 4: {e}")

    return transactions


# ---------------------------------------------------------------------------
# 8-K — Material Event Detection
# ---------------------------------------------------------------------------

NEGATIVE_8K_KEYWORDS = [
    "restatement", "restate", "material weakness",
    "investigation", "sec inquiry", "subpoena", "grand jury",
    "resignation", "terminated", "departure",
    "going concern", "bankruptcy", "default", "liquidity",
]


def is_negative_8k(cik: str, accession_number: str) -> bool:
    """
    Detect negative 8-K filings by scanning the document for red-flag keywords.

    8-K item numbers that indicate problems:
    - Item 4.02: Non-reliance on previously issued financial statements
    - Item 5.02: Departure of directors or officers
    - Item 8.01: Other events (catch-all — need keyword scan)

    We fetch the filing index page and scan the full text for keywords.
    This is simpler than parsing the 8-K XML structure.
    """
    cik_int = str(int(cik.lstrip("0") or "0"))
    accession_clean = accession_number.replace("-", "")
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{accession_clean}/{accession_number}-index.htm"
    )
    try:
        content = _get(index_url).text.lower()
        return any(kw in content for kw in NEGATIVE_8K_KEYWORDS)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def store_filing(stock_id: int, accession_number: str, form_type: str,
                 filed_date: str, period: str, raw_json: str) -> Optional[int]:
    """
    Insert a filing record. Returns new row ID, or None if already exists.
    INSERT OR IGNORE means duplicate accession numbers are silently skipped.
    """
    try:
        with get_db() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO sec_filings
                   (stock_id, accession_number, form_type, filed_date,
                    period_of_report, raw_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (stock_id, accession_number, form_type,
                 filed_date, period, raw_json),
            )
            return cursor.lastrowid if cursor.rowcount > 0 else None
    except Exception as e:
        print(f"    Error storing filing {accession_number}: {e}")
        return None


def store_insider_transactions(stock_id: int, filing_id: int,
                                transactions: list[dict]):
    if not transactions:
        return
    rows = [
        (
            stock_id, filing_id,
            t["transaction_date"], t["insider_name"], t["insider_title"],
            t["transaction_type"], t["shares"], t["price_per_share"],
            t["total_value"], t["shares_owned_after"],
        )
        for t in transactions
    ]
    executemany(
        """INSERT INTO insider_transactions
           (stock_id, filing_id, transaction_date, insider_name, insider_title,
            transaction_type, shares, price_per_share, total_value, shares_owned_after)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_stock(stock_id: int, ticker: str, cik: str,
                  days_back: int = 90, as_of_date: str = None):
    """
    Fetch and process all recent SEC filings for one stock.
    Called by data/pipeline.py for each stock in the universe.
    as_of_date: anchor for the lookback window — pass for historical reruns.
    """
    if not cik:
        print(f"  {ticker}: no CIK — skipping SEC collection")
        return

    print(f"  {ticker}: fetching SEC filings (last {days_back} days)...")

    try:
        filings = get_recent_filings(cik, SEC["target_form_types"], days_back, as_of_date)
    except Exception as e:
        print(f"  {ticker}: error fetching filing history — {e}")
        return

    if not filings:
        print(f"  {ticker}: no new filings")
        return

    for filing in filings:
        accession  = filing["accessionNumber"]
        form_type  = filing["form"]
        filed_date = filing["filingDate"]
        period     = filing.get("reportDate", "")
        primary    = filing.get("primaryDocument", "")

        filing_id = store_filing(
            stock_id, accession, form_type,
            filed_date, period, json.dumps(filing)
        )

        if filing_id is None:
            continue  # Already processed

        if form_type == "4" and primary:
            xml = fetch_form4_xml(cik, accession, primary)
            if xml:
                transactions = parse_form4(xml)
                store_insider_transactions(stock_id, filing_id, transactions)
                if transactions:
                    buys  = sum(1 for t in transactions if t["transaction_type"] == "P")
                    sells = sum(1 for t in transactions if t["transaction_type"] == "S")
                    print(f"    Form 4: {buys} buys, {sells} sells")

        elif form_type == "8-K":
            negative = is_negative_8k(cik, accession)
            flag = "⚠️  NEGATIVE" if negative else "neutral"
            print(f"    8-K ({filed_date}): {flag}")
            # Persist negativity flag so sentiment feature store can query it
            with get_db() as conn:
                conn.execute(
                    "UPDATE sec_filings SET is_negative_8k = ? WHERE id = ?",
                    (int(negative), filing_id)
                )

        elif form_type in ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"):
            print(f"    {form_type} ({filed_date}): activist/institutional filing")

        # Mark processed
        with get_db() as conn:
            conn.execute(
                "UPDATE sec_filings SET processed = 1 WHERE id = ?",
                (filing_id,)
            )
