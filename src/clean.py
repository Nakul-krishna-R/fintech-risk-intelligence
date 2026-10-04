"""Clean raw 8-K filing text and chunk it for downstream embedding/analysis.

Reads every filing listed in data/raw/filings_metadata.csv (produced by collect.py),
strips the standard SEC cover-page boilerplate and signature block, normalizes
whitespace, splits what's left into ~500-token chunks with 50-token overlap, and
writes one row per chunk to:

    data/cleaned/chunks.csv   columns: company, cik, filing_date, chunk_id, chunk_text

Token counts use tiktoken's cl100k_base encoding when available (falls back to a
whitespace-word approximation, with a warning, if tiktoken isn't installed).

Usage:
    python src/clean.py
    python src/clean.py --chunk-size 400 --overlap 40
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def encode(text: str) -> list[int]:
        return _ENCODER.encode(text)

    def decode(tokens: list[int]) -> str:
        return _ENCODER.decode(tokens)

except ImportError:
    print(
        "Warning: tiktoken not installed; falling back to a whitespace-word token "
        "approximation (pip install tiktoken for accurate counts).",
        file=sys.stderr,
    )

    def encode(text: str) -> list[str]:
        return text.split()

    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


REPO_ROOT = Path(__file__).resolve().parents[1]

# The cover page (registrant identity, checkboxes, registered-securities table) is
# identical in shape across every 8-K and carries no filing-specific content; it ends
# right where the first numbered item begins.
HEADER_END_RE = re.compile(r"(?m)^Item\s+\d+\.\d+\b")

# The signature block (name/title/page-number footer) is equally formulaic and adds
# nothing but a person's name, so it's dropped along with the header.
SIGNATURE_RE = re.compile(r"(?m)^SIGNATURES?\s*$")

BLANK_LINES_RE = re.compile(r"\n\s*\n\s*\n+")
TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")


def strip_boilerplate(text: str) -> str:
    """Drop the SEC cover-page header and signature-block footer."""
    header_match = HEADER_END_RE.search(text)
    if header_match:
        text = text[header_match.start():]

    signature_match = SIGNATURE_RE.search(text)
    if signature_match:
        text = text[: signature_match.start()]

    return text


def normalize_whitespace(text: str) -> str:
    text = TRAILING_SPACE_RE.sub("\n", text)
    text = BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(raw: str) -> str:
    return normalize_whitespace(strip_boilerplate(raw))


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping ~chunk_size-token windows."""
    tokens = encode(text)
    if not tokens:
        return []

    stride = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += stride
    return chunks


def load_metadata(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def normalize_company_names(rows: list[dict]) -> None:
    """Collapse case-only company-name variants for the same CIK in place.

    EDGAR's own metadata is inconsistent about casing across filings (e.g. a CIK
    shows up as both "GENERAL ELECTRIC Co" and "GENERAL ELECTRIC CO"), which would
    otherwise split one company into two groups downstream. Genuine historical
    renames (e.g. Facebook Inc -> Meta Platforms, Inc.) differ by more than case and
    are left untouched.
    """
    from collections import Counter, defaultdict

    variants = defaultdict(Counter)
    for row in rows:
        variants[(row["cik"], row["company"].casefold())][row["company"]] += 1

    canonical = {key: counts.most_common(1)[0][0] for key, counts in variants.items()}
    for row in rows:
        row["company"] = canonical[(row["cik"], row["company"].casefold())]


def process(args: argparse.Namespace) -> int:
    metadata_path = Path(args.metadata)
    filings_dir = Path(args.filings_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_metadata(metadata_path)
    print(f"Loaded {len(rows)} filings from {metadata_path}")

    out_rows = []
    missing, empty = 0, 0

    for index, row in enumerate(rows, start=1):
        ticker = row["ticker"]
        filing_date = row["filing_date"]
        text_path = filings_dir / Path(row["text_file"]).name

        if not text_path.exists():
            print(f"[{index}/{len(rows)}] {ticker} {filing_date}: MISSING {text_path.name}")
            missing += 1
            continue

        raw = text_path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned, args.chunk_size, args.overlap)

        if not chunks:
            print(f"[{index}/{len(rows)}] {ticker} {filing_date}: empty after cleaning, skipped")
            empty += 1
            continue

        for i, chunk in enumerate(chunks):
            out_rows.append(
                {
                    "company": row["company_name"],
                    "cik": row["cik"],
                    "filing_date": filing_date,
                    "chunk_id": f"{ticker}_{row['accession_number']}_chunk{i:03d}",
                    "chunk_text": chunk,
                }
            )

        if index % 50 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] {ticker} {filing_date}: {len(chunks)} chunks "
                  f"({len(out_rows)} total so far)")

    normalize_company_names(out_rows)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["company", "cik", "filing_date", "chunk_id", "chunk_text"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nDone: {len(out_rows)} chunks from {len(rows) - missing - empty} filings -> {out_path}")
    if missing:
        print(f"  {missing} filing(s) had no matching .txt file")
    if empty:
        print(f"  {empty} filing(s) were empty after cleaning")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--metadata",
        default=REPO_ROOT / "data" / "raw" / "filings_metadata.csv",
        help="path to filings_metadata.csv (default: data/raw/filings_metadata.csv)",
    )
    parser.add_argument(
        "--filings-dir",
        default=REPO_ROOT / "data" / "raw" / "filings",
        help="directory of raw filing .txt files (default: data/raw/filings)",
    )
    parser.add_argument(
        "--out",
        default=REPO_ROOT / "data" / "cleaned" / "chunks.csv",
        help="output CSV path (default: data/cleaned/chunks.csv)",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="tokens per chunk (default: 500)")
    parser.add_argument("--overlap", type=int, default=50, help="token overlap between chunks (default: 50)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(process(parse_args()))
