"""Semantic retrieval over embedded 8-K chunks, with same-filing leakage filtering.

`get_similar_filings(chunk_id)` embeds the given chunk, pulls its nearest neighbours
out of the ChromaDB collection built by embed.py, discards neighbours that came from
the same filing (same company on the same filing date) along with the chunk itself,
and summarizes what's left into a small feature dict.

Dropping same-filing neighbours matters because clean.py chunks overlap by 50 tokens,
so a chunk's nearest neighbours are often its own siblings. Keeping them would let a
filing describe itself, which leaks straight into any model trained on these features.

Usage:
    from retrieve import get_similar_filings
    features = get_similar_filings("AAPL_0000320193-21-000009_chunk000")

    python src/retrieve.py              # smoke test on 5 random chunks
    python src/retrieve.py --seed 42    # reproducible sample
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHROMA_DIR = REPO_ROOT / "data" / "cleaned" / "chromadb"
EXTRACTIONS_CSV = REPO_ROOT / "data" / "cleaned" / "extractions.csv"
COLLECTION_NAME = "filing_chunks"
MODEL_NAME = "all-mpnet-base-v2"

# Loading the model costs ~15s and the client re-opens the DB, so both are cached at
# module level — callers that score thousands of chunks pay that cost once.
_model = None
_collection = None


def get_model(model_name: str = MODEL_NAME, device: str = "auto"):
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(model_name, device=device)
    return _model


def get_collection(chroma_dir: Path = CHROMA_DIR, collection: str = COLLECTION_NAME):
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        _collection = client.get_collection(collection)
    return _collection


def get_similar_filings(
    chunk_id: str,
    top_k: int = 5,
    exclude_same_company: bool = False,
    n_candidates: int = 50,
) -> dict:
    """Summarize the nearest neighbours of `chunk_id`, excluding its own filing.

    Pulls `n_candidates` nearest neighbours, drops the ones the leakage filters
    reject, then keeps the top `top_k` survivors. Over-fetching matters because 8-K
    language is highly company-specific: a chunk's closest matches are usually other
    filings by the same issuer, so a candidate pool of only `top_k` leaves nothing
    behind once same-company neighbours are removed.

    Args:
        chunk_id: id of the chunk to look up (must exist in the collection).
        top_k: how many surviving neighbours to summarize.
        exclude_same_company: if True, drop every neighbour from the same company
            rather than only those from the same company *and* filing date. Stricter
            leakage guard for company-level modelling.
        n_candidates: size of the candidate pool fetched before filtering.

    Returns:
        dict with:
            similar_count             neighbours left after filtering (0..top_k)
            avg_severity              mean severity of those neighbours
            dominant_risk_category    most common risk_category among them
            negative_sentiment_ratio  share of them with sentiment == "negative"

        The three aggregates are None when similar_count is 0, so callers can tell
        "no valid neighbours" apart from a genuine zero.
    """
    collection = get_collection()

    source = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if not source["ids"]:
        raise ValueError(f"chunk_id {chunk_id!r} not found in collection {COLLECTION_NAME!r}")

    source_meta = source["metadatas"][0]
    source_text = source["documents"][0]

    # Re-embed the chunk text rather than reusing the stored vector, so this works
    # for any text and stays honest about which model drives retrieval.
    model = get_model()
    query_vector = model.encode([source_text], normalize_embeddings=True)[0]

    # Fetch a pool well beyond top_k so the filters below have something to keep.
    # +1 covers the chunk itself, which is always its own closest match.
    pool_size = min(max(n_candidates, top_k + 1), collection.count())
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=pool_size,
        include=["metadatas"],
    )
    neighbour_ids = result["ids"][0]
    neighbour_metas = result["metadatas"][0]

    kept = []
    for neighbour_id, meta in zip(neighbour_ids, neighbour_metas):
        if neighbour_id == chunk_id:
            continue
        same_company = meta["company"] == source_meta["company"]
        if exclude_same_company and same_company:
            continue
        if same_company and meta["filing_date"] == source_meta["filing_date"]:
            continue
        kept.append(meta)
        if len(kept) == top_k:
            break

    if not kept:
        return {
            "similar_count": 0,
            "avg_severity": None,
            "dominant_risk_category": None,
            "negative_sentiment_ratio": None,
        }

    severities = [m["severity"] for m in kept]
    categories = Counter(m["risk_category"] for m in kept)
    negatives = sum(1 for m in kept if m["sentiment"] == "negative")

    return {
        "similar_count": len(kept),
        "avg_severity": sum(severities) / len(severities),
        "dominant_risk_category": categories.most_common(1)[0][0],
        "negative_sentiment_ratio": negatives / len(kept),
    }


def _smoke_test(seed: int | None, sample_size: int = 5) -> int:
    """Run get_similar_filings on a few random chunk_ids and print the results."""
    with EXTRACTIONS_CSV.open(newline="", encoding="utf-8") as f:
        chunk_ids = [row["chunk_id"] for row in csv.DictReader(f)]

    if seed is not None:
        random.seed(seed)
    sample = random.sample(chunk_ids, min(sample_size, len(chunk_ids)))

    print(f"Testing get_similar_filings on {len(sample)} random chunk_ids\n")
    for chunk_id in sample:
        features = get_similar_filings(chunk_id)
        print(chunk_id)
        for key, value in features.items():
            if isinstance(value, float):
                print(f"  {key:26} {value:.3f}")
            else:
                print(f"  {key:26} {value}")
        print()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=None, help="seed the random sample for reproducibility")
    parser.add_argument("--sample-size", type=int, default=5, help="how many chunk_ids to test (default: 5)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    sys.exit(_smoke_test(args.seed, args.sample_size))
