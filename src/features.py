"""Build the filing-level feature matrix for modelling.

Three stages:

1. Aggregate chunk-level rows in extractions.csv up to one row per filing
   (keyed on accession number, parsed out of chunk_id).
2. Call retrieve.get_similar_filings on each filing's first chunk to attach
   neighbourhood features from the embedding index.
3. Pull daily prices from yfinance and label the outcome: stock_drop == 1 when the
   stock fell more than 10% within 90 days of the filing date.

Output: data/features/feature_matrix.csv

The outcome definition is switchable:
    --drop-method trough   (default) 1 if the lowest close in the window is >10%
                           below the baseline, i.e. the drop happened at any point
    --drop-method return   1 if the close at the end of the window is >10% below
                           the baseline, i.e. a point-to-point 90-day return

Baseline is the close on the first trading day on or after the filing date.

Usage:
    python src/features.py
    python src/features.py --drop-method return
    python src/features.py --limit 50          # quick test run
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

DROP_THRESHOLD = -0.10
WINDOW_DAYS = 90

OUT_FIELDS = [
    # identity
    "accession_number", "ticker", "company", "cik", "filing_date",
    # filing-level aggregates
    "dominant_risk_category", "avg_severity", "negative_sentiment_ratio",
    "forward_looking_ratio", "chunk_count", "parse_error_count",
    # retrieval features
    "similar_count", "avg_severity_neighbours",
    "dominant_risk_category_neighbours", "negative_sentiment_ratio_neighbours",
    # outcome and the prices behind it
    "baseline_price", "min_price_90d", "max_drawdown_90d", "fwd_return_90d",
    "stock_drop",
]


def parse_chunk_id(chunk_id: str) -> tuple[str, str, int]:
    """Split 'AAPL_0001193125-21-001982_chunk000' into (ticker, accession, index)."""
    base, chunk_part = chunk_id.rsplit("_", 1)
    ticker, accession = base.split("_", 1)
    return ticker, accession, int(chunk_part.replace("chunk", ""))


def aggregate_filings(extractions_path: Path) -> list[dict]:
    """Collapse chunk rows into one record per filing (accession number)."""
    with extractions_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        ticker, accession, index = parse_chunk_id(row["chunk_id"])
        row["_ticker"] = ticker
        row["_index"] = index
        grouped.setdefault(accession, []).append(row)

    filings = []
    for accession, chunks in grouped.items():
        chunks.sort(key=lambda r: r["_index"])
        first = chunks[0]

        severities = []
        for chunk in chunks:
            try:
                severities.append(int(chunk["severity"]))
            except (TypeError, ValueError):
                pass

        negatives = sum(1 for c in chunks if c["sentiment"] == "negative")
        forward = sum(1 for c in chunks if c["contains_forward_looking_statement"].strip().lower() == "true")
        categories = Counter(c["risk_category"] for c in chunks)

        filings.append(
            {
                "accession_number": accession,
                "ticker": first["_ticker"],
                "company": first["company"],
                "cik": first["cik"],
                "filing_date": first["filing_date"],
                "dominant_risk_category": categories.most_common(1)[0][0],
                "avg_severity": sum(severities) / len(severities) if severities else None,
                "negative_sentiment_ratio": negatives / len(chunks),
                "forward_looking_ratio": forward / len(chunks),
                "chunk_count": len(chunks),
                # carried through so modelling can control for low-confidence extractions
                "parse_error_count": sum(1 for c in chunks if c["parse_error"]),
                "_first_chunk_id": first["chunk_id"],
            }
        )

    filings.sort(key=lambda r: (r["ticker"], r["filing_date"]))
    return filings


def add_retrieval_features(
    filings: list[dict], exclude_same_company: bool, n_candidates: int
) -> None:
    """Attach neighbourhood features from the embedding index, in place."""
    from retrieve import get_similar_filings

    print(f"\nRetrieving neighbours for {len(filings)} filings "
          f"(candidate pool={n_candidates}, exclude_same_company={exclude_same_company})...")
    failures = 0

    for i, filing in enumerate(filings, start=1):
        try:
            neighbours = get_similar_filings(
                filing["_first_chunk_id"],
                exclude_same_company=exclude_same_company,
                n_candidates=n_candidates,
            )
            filing["similar_count"] = neighbours["similar_count"]
            filing["avg_severity_neighbours"] = neighbours["avg_severity"]
            filing["dominant_risk_category_neighbours"] = neighbours["dominant_risk_category"]
            filing["negative_sentiment_ratio_neighbours"] = neighbours["negative_sentiment_ratio"]
        except Exception as exc:
            failures += 1
            if failures <= 3:
                print(f"  {filing['_first_chunk_id']}: retrieval failed ({exc})")
            filing["similar_count"] = None
            filing["avg_severity_neighbours"] = None
            filing["dominant_risk_category_neighbours"] = None
            filing["negative_sentiment_ratio_neighbours"] = None

        if i % 100 == 0 or i == len(filings):
            print(f"  [{i}/{len(filings)}] neighbours retrieved")

    if failures:
        print(f"  {failures} filing(s) had no retrieval features")


def fetch_prices(tickers: list[str], start: str, end: str, delay: float) -> dict:
    """Download daily closes once per ticker, not once per filing."""
    import yfinance as yf

    print(f"\nDownloading prices for {len(tickers)} tickers ({start} -> {end})...")
    prices = {}

    for i, ticker in enumerate(sorted(tickers), start=1):
        try:
            history = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
        except Exception as exc:
            print(f"  [{i}/{len(tickers)}] {ticker}: download failed ({exc})")
            continue

        if history.empty:
            print(f"  [{i}/{len(tickers)}] {ticker}: no price data")
            continue

        closes = history["Close"]
        closes.index = closes.index.tz_localize(None).normalize()
        prices[ticker] = closes
        print(f"  [{i}/{len(tickers)}] {ticker}: {len(closes)} trading days")
        time.sleep(delay)

    return prices


def label_outcomes(filings: list[dict], prices: dict, drop_method: str) -> None:
    """Compute the 90-day price outcome and stock_drop flag, in place."""
    print(f"\nLabelling outcomes (method={drop_method}, threshold={DROP_THRESHOLD:.0%})...")
    missing = 0

    for filing in filings:
        for field in ("baseline_price", "min_price_90d", "max_drawdown_90d",
                      "fwd_return_90d", "stock_drop"):
            filing[field] = None

        closes = prices.get(filing["ticker"])
        if closes is None:
            missing += 1
            continue

        filed = datetime.strptime(filing["filing_date"], "%Y-%m-%d")

        # Baseline: first trading day on or after the filing date.
        on_or_after = closes[closes.index >= filed]
        if on_or_after.empty:
            missing += 1
            continue
        baseline_date = on_or_after.index[0]
        baseline = float(on_or_after.iloc[0])

        window = closes[(closes.index > baseline_date) & (closes.index <= filed + timedelta(days=WINDOW_DAYS))]
        if window.empty:
            missing += 1
            continue

        min_price = float(window.min())
        drawdown = (min_price - baseline) / baseline
        fwd_return = (float(window.iloc[-1]) - baseline) / baseline

        measure = drawdown if drop_method == "trough" else fwd_return

        filing["baseline_price"] = round(baseline, 4)
        filing["min_price_90d"] = round(min_price, 4)
        filing["max_drawdown_90d"] = round(drawdown, 6)
        filing["fwd_return_90d"] = round(fwd_return, 6)
        filing["stock_drop"] = int(measure < DROP_THRESHOLD)

    if missing:
        print(f"  {missing} filing(s) had no usable price window; stock_drop left blank")


def build(args: argparse.Namespace) -> int:
    extractions_path = Path(args.extractions)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    filings = aggregate_filings(extractions_path)
    print(f"Aggregated {len(filings)} filings from {extractions_path}")
    if args.limit:
        filings = filings[: args.limit]
        print(f"  limited to first {len(filings)}")

    add_retrieval_features(filings, args.exclude_same_company, args.n_candidates)

    # Pad the price window so the last filing still has 90 days of history ahead.
    dates = [f["filing_date"] for f in filings]
    start = (datetime.strptime(min(dates), "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (datetime.strptime(max(dates), "%Y-%m-%d") + timedelta(days=WINDOW_DAYS + 14)).strftime("%Y-%m-%d")

    prices = fetch_prices(sorted({f["ticker"] for f in filings}), start, end, args.delay)
    label_outcomes(filings, prices, args.drop_method)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filings)

    labelled = [f for f in filings if f["stock_drop"] is not None]
    positives = sum(f["stock_drop"] for f in labelled)
    print(f"\nDone: {len(filings)} filings -> {out_path}")
    print(f"  labelled: {len(labelled)} | stock_drop=1: {positives} "
          f"({positives / len(labelled):.1%} positive rate)" if labelled else "  no labelled rows")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--extractions",
        default=REPO_ROOT / "data" / "cleaned" / "extractions.csv",
        help="input extractions CSV (default: data/cleaned/extractions.csv)",
    )
    parser.add_argument(
        "--out",
        default=REPO_ROOT / "data" / "features" / "feature_matrix.csv",
        help="output CSV path (default: data/features/feature_matrix.csv)",
    )
    parser.add_argument(
        "--drop-method",
        default="trough",
        choices=["trough", "return"],
        help="how to measure the 90-day drop (default: trough)",
    )
    parser.add_argument(
        "--exclude-same-company",
        action="store_true",
        help="stricter retrieval filter: drop all same-company neighbours",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=50,
        help="neighbour candidate pool fetched before filtering (default: 50)",
    )
    parser.add_argument("--delay", type=float, default=0.3, help="seconds between yfinance calls (default: 0.3)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N filings (testing)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(build(parse_args()))
