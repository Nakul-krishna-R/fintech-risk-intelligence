"""Collect 8-K filings for 50 large US companies from SEC EDGAR full-text search.

For every company the script queries EDGAR full-text search for 8-K filings in the
requested date range, downloads each filing's primary document, strips it to plain
text and writes:

    data/raw/filings_metadata.csv   one row per filing
    data/raw/filings/*.txt          the raw text of each filing

The SEC asks that automated clients identify themselves and stay under 10 requests
per second. Set SEC_USER_AGENT (or pass --user-agent) to "Name email@example.com"
and leave --delay at its default.

Usage:
    python src/collect.py
    python src/collect.py --start 2021-01-01 --end 2024-12-31 --max-per-company 20
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, asdict, fields
from pathlib import Path

try:
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
except ImportError:  # pragma: no cover - dependency check
    sys.exit("collect.py requires beautifulsoup4 and lxml:\n    pip install beautifulsoup4 lxml")

# Inline-XBRL filings open with an XML declaration; parsing them as HTML is what we
# want (we are after the rendered text), so silence bs4's warning about it.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


FTS_URL = "https://efts.sec.gov/LATEST/search-index"
TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

DEFAULT_USER_AGENT = "fintech-risk-intelligence nakulkrishna96@gmail.com"

# 50 large US companies by market capitalisation. Tickers are resolved to CIK
# numbers at runtime against the SEC's own ticker file, so this list stays readable.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "AVGO", "LLY",
    "JPM", "V", "XOM", "UNH", "MA", "COST", "HD", "PG", "JNJ", "ABBV",
    "WMT", "MRK", "NFLX", "KO", "ADBE", "PEP", "CVX", "CRM", "AMD", "TMO",
    "BAC", "LIN", "CSCO", "ACN", "MCD",  "ABT", "PM", "ORCL", "IBM", "GE",
    "INTC", "DIS", "QCOM", "TXN", "CAT", "VZ", "AMGN", "PFE", "NKE", "WFC",
]

# The SEC ticker file points at the *current* registrant, which is not always the
# entity that filed historically. XOM now resolves to ExxonMobil Holdings Corp
# (CIK 2115436), a recently registered holding company with no filings before 2025;
# the 2021-2024 8-Ks sit under Exxon Mobil Corp.
CIK_OVERRIDES = {
    "XOM": ("0000034088", "EXXON MOBIL CORP"),
}


@dataclass
class Filing:
    """One 8-K filing and the location of its extracted text."""

    ticker: str
    company_name: str
    cik: str
    filing_date: str
    accession_number: str
    form_type: str
    items: str
    primary_document: str
    filing_url: str
    text_file: str
    text_length: int


CSV_COLUMNS = [f.name for f in fields(Filing)]


class Client:
    """Minimal HTTP client that keeps a fixed delay between requests and retries."""

    def __init__(self, user_agent: str, delay: float, retries: int = 3):
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self.delay = delay
        self.retries = retries
        self._last_request = 0.0

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.monotonic()

    def get(self, url: str, params: dict | None = None) -> bytes:
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers=self.headers)

        for attempt in range(1, self.retries + 1):
            self._wait()
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                    if response.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    raise RuntimeError(
                        "EDGAR returned 403 Forbidden. Set a contact User-Agent via "
                        "SEC_USER_AGENT or --user-agent, e.g. 'Jane Doe jane@example.com'."
                    ) from exc
                if exc.code == 404 or exc.code < 500 and exc.code != 429:
                    raise   
                last = exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc

            if attempt < self.retries:
                backoff = self.delay * 2**attempt
                print(f"    retry {attempt}/{self.retries - 1} in {backoff:.1f}s ({last})")
                time.sleep(backoff)

        raise RuntimeError(f"giving up on {url}: {last}")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        return json.loads(self.get(url, params))


def load_cik_map(client: Client) -> dict[str, tuple[str, str]]:
    """Map ticker -> (zero-padded CIK, company name) using the SEC ticker file."""
    data = client.get_json(TICKER_URL)
    cik_map = {
        entry["ticker"]: (str(entry["cik_str"]).zfill(10), entry["title"])
        for entry in data.values()
    }
    cik_map.update(CIK_OVERRIDES)
    return cik_map


def search_8k(client: Client, cik: str, start: str, end: str, max_hits: int) -> list[dict]:
    """Return full-text-search hits for a company's 8-K filings in a date range.

    The `q` parameter is deliberately omitted: with a keyword the API matches
    individual documents (including exhibits), while without one it returns a
    single hit per filing, which is what we want here.
    """
    hits: list[dict] = []
    offset = 0

    while len(hits) < max_hits:
        payload = client.get_json(
            FTS_URL,
            {
                "forms": "8-K",
                "ciks": cik,
                "startdt": start,
                "enddt": end,
                "from": offset,
            },
        )
        page = payload.get("hits", {}).get("hits", [])
        if not page:
            break
        hits.extend(page)
        offset += len(page)

        total = payload.get("hits", {}).get("total", {}).get("value", 0)
        if offset >= total or offset >= 10_000:  # EDGAR caps deep paging
            break

    return hits[:max_hits]


def html_to_text(payload: bytes, filename: str) -> str:
    """Strip an EDGAR primary document down to readable text."""
    if filename.lower().endswith((".txt", ".xml")):
        text = payload.decode("utf-8", errors="replace")
    else:
        soup = BeautifulSoup(payload, "lxml")
        # Modern filings are inline XBRL: <ix:header> holds a block of hidden facts
        # (context ids, tagged dates, member names) that is not part of the document
        # a reader sees, so it would otherwise land at the top of every text file.
        for tag in soup(["script", "style", "head", "ix:header"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def load_done(csv_path: Path) -> tuple[list[dict], set[str]]:
    """Read an existing metadata CSV so an interrupted run can be resumed."""
    if not csv_path.exists():
        return [], set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows, {row["accession_number"] for row in rows}


def collect(args: argparse.Namespace) -> int:
    client = Client(args.user_agent, args.delay)

    metadata_path = Path(args.out_dir) / "filings_metadata.csv"
    filings_dir = Path(args.out_dir) / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)

    rows, done = ([], set()) if args.no_resume else load_done(metadata_path)
    if done:
        print(f"Resuming: {len(done)} filings already collected")

    tickers = args.tickers
    print(f"Resolving {len(tickers)} tickers to CIK numbers...")
    cik_map = load_cik_map(client)
    unknown = [t for t in tickers if t not in cik_map]
    if unknown:
        print(f"Warning: no CIK found for {', '.join(unknown)}")

    for index, ticker in enumerate(tickers, start=1):
        if ticker not in cik_map:
            continue
        cik, company_name = cik_map[ticker]
        print(f"[{index}/{len(tickers)}] {ticker} ({company_name}, CIK {cik})")

        try:
            hits = search_8k(client, cik, args.start, args.end, args.max_per_company)
        except Exception as exc:  # keep going; one bad company shouldn't end the run
            print(f"    search failed: {exc}")
            continue
        if not hits:
            # Usually means the ticker now resolves to a successor entity that did
            # not file in this window - see CIK_OVERRIDES.
            print(f"    no filings found for CIK {cik}; check the ticker-to-CIK mapping")
        else:
            print(f"    {len(hits)} filings found")

        for hit in hits:
            source = hit["_source"]
            accession = source["adsh"]
            if accession in done:
                continue

            # _id is "<accession>:<primary document filename>"
            primary_document = hit["_id"].split(":", 1)[1]
            url = (
                f"{ARCHIVES_URL}/{int(cik)}/{accession.replace('-', '')}/{primary_document}"
            )

            try:
                text = html_to_text(client.get(url), primary_document)
            except Exception as exc:
                print(f"    {accession}: download failed: {exc}")
                continue

            text_file = f"{ticker}_{cik}_{source['file_date']}_{accession}.txt"
            (filings_dir / text_file).write_text(text, encoding="utf-8")

            rows.append(
                asdict(
                    Filing(
                        ticker=ticker,
                        company_name=source["display_names"][0].split("  (")[0],
                        cik=cik,
                        filing_date=source["file_date"],
                        accession_number=accession,
                        form_type=source.get("form", "8-K"),
                        items="; ".join(source.get("items") or []),
                        primary_document=primary_document,
                        filing_url=url,
                        text_file=f"filings/{text_file}",
                        text_length=len(text),
                    )
                )
            )
            done.add(accession)

        write_metadata(metadata_path, rows)  # checkpoint after each company

    print(f"\nDone: {len(rows)} filings -> {metadata_path}")
    return 0


def write_metadata(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["ticker"], r["filing_date"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2021-01-01", help="earliest filing date")
    parser.add_argument("--end", default="2024-12-31", help="latest filing date")
    parser.add_argument(
        "--out-dir",
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="output directory (default: data/raw)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5, help="seconds between requests (default: 0.5)"
    )
    parser.add_argument(
        "--max-per-company",
        type=int,
        default=200,
        help="cap on filings per company (default: 200)",
    )
    parser.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT),
        help="contact string sent to the SEC; also read from SEC_USER_AGENT",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
        metavar="TICKER",
        help="override the built-in company list (useful for a quick test run)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing metadata CSV and start over",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:    
        sys.exit(collect(parse_args()))
    except KeyboardInterrupt:
        sys.exit("\nInterrupted; rerun to resume from the existing CSV.")
