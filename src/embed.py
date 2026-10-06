"""Embed filing chunks with all-mpnet-base-v2 and store them in ChromaDB.

Joins data/cleaned/chunks.csv with data/cleaned/extractions.csv on chunk_id, embeds
each chunk's text on the GPU, and writes vectors plus metadata to a persistent
ChromaDB collection at data/cleaned/chromadb/.

Metadata stored per vector:
    chunk_id, company, cik, filing_date, risk_category, sentiment, severity

Note: all-mpnet-base-v2 has a 384-wordpiece input limit, so the tail of a ~500-token
chunk is truncated by the model. That's a property of the model, not of this script.

Usage:
    python src/embed.py
    python src/embed.py --limit 200           # quick test run
    python src/embed.py --reset               # drop the collection and re-embed
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "all-mpnet-base-v2"
COLLECTION_NAME = "filing_chunks"

# Only these columns ride along into ChromaDB; everything else in the CSVs is
# derivable from chunk_id if it's needed later.
METADATA_FIELDS = [
    "chunk_id", "company", "cik", "filing_date",
    "risk_category", "sentiment", "severity",
]


def load_joined(chunks_path: Path, extractions_path: Path) -> list[dict]:
    """Inner-join chunks and extractions on chunk_id, preserving chunk order."""
    with extractions_path.open(newline="", encoding="utf-8") as f:
        extractions = {row["chunk_id"]: row for row in csv.DictReader(f)}

    with chunks_path.open(newline="", encoding="utf-8") as f:
        chunks = list(csv.DictReader(f))

    joined, unmatched = [], 0
    for chunk in chunks:
        extraction = extractions.get(chunk["chunk_id"])
        if extraction is None:
            unmatched += 1
            continue

        # severity goes in as an int so Chroma can filter it numerically ($gte etc.)
        try:
            severity = int(extraction["severity"])
        except (TypeError, ValueError):
            severity = 0

        joined.append(
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["chunk_text"],
                "metadata": {
                    "chunk_id": chunk["chunk_id"],
                    "company": chunk["company"],
                    "cik": chunk["cik"],
                    "filing_date": chunk["filing_date"],
                    "risk_category": extraction["risk_category"],
                    "sentiment": extraction["sentiment"],
                    "severity": severity,
                },
            }
        )

    if unmatched:
        print(f"Warning: {unmatched} chunk(s) had no matching row in extractions.csv and were skipped")
    return joined


def build_model(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    print(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    # Renamed in sentence-transformers 6; keep working on older versions too.
    get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    print(f"  embedding dim: {get_dim()}, max input: {model.max_seq_length} wordpieces")
    return model


def embed(args: argparse.Namespace) -> int:
    import chromadb

    records = load_joined(Path(args.chunks), Path(args.extractions))
    if args.limit:
        records = records[: args.limit]
    print(f"Joined {len(records)} chunks")
    if not records:
        print("Nothing to embed.")
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(out_dir),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Dropped existing collection {args.collection!r}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine", "embedding_model": args.model},
    )

    # Skip anything already stored so an interrupted run can be picked up again.
    existing = set()
    if not args.reset and collection.count():
        existing = set(collection.get(include=[])["ids"])
        print(f"Collection already holds {len(existing)} vectors; those will be skipped")

    pending = [r for r in records if r["chunk_id"] not in existing]
    if not pending:
        print(f"\nDone: nothing new to embed; collection holds {collection.count()} vectors")
        return 0

    model = build_model(args.model, args.device)

    done = 0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]

        vectors = model.encode(
            [r["text"] for r in batch],
            batch_size=args.encode_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        collection.add(
            ids=[r["chunk_id"] for r in batch],
            documents=[r["text"] for r in batch],
            embeddings=[v.tolist() for v in vectors],
            metadatas=[r["metadata"] for r in batch],
        )

        done += len(batch)
        if done % 100 == 0 or done == len(pending):
            print(f"[{done}/{len(pending)}] embedded")

    print(f"\nDone: {done} chunks embedded -> {out_dir}")
    print(f"  collection {args.collection!r} now holds {collection.count()} vectors")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--chunks",
        default=REPO_ROOT / "data" / "cleaned" / "chunks.csv",
        help="input chunks CSV (default: data/cleaned/chunks.csv)",
    )
    parser.add_argument(
        "--extractions",
        default=REPO_ROOT / "data" / "cleaned" / "extractions.csv",
        help="input extractions CSV (default: data/cleaned/extractions.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default=REPO_ROOT / "data" / "cleaned" / "chromadb",
        help="ChromaDB persist directory (default: data/cleaned/chromadb)",
    )
    parser.add_argument("--collection", default=COLLECTION_NAME, help=f"collection name (default: {COLLECTION_NAME})")
    parser.add_argument("--model", default=MODEL_NAME, help=f"sentence-transformers model (default: {MODEL_NAME})")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="compute device (default: auto)")
    parser.add_argument("--batch-size", type=int, default=100, help="chunks added to Chroma per batch (default: 100)")
    parser.add_argument("--encode-batch-size", type=int, default=32, help="GPU encode batch size (default: 32)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N chunks (testing)")
    parser.add_argument("--reset", action="store_true", help="drop the collection and re-embed from scratch")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(embed(parse_args()))
