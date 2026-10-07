"""FastAPI service that scores a raw 8-K filing through the risk pipeline.

The request path reuses the pipeline modules directly rather than reimplementing
them, so a scored filing goes through the same cleaning, chunking, extraction,
embedding and aggregation steps the training data did:

    clean.py     clean_text / chunk_text
    extract.py   OllamaClient / parse_extraction  (local phi3:mini)
    retrieve.py  cached embedder + ChromaDB collection
    features.py  the filing-level aggregation this mirrors

Run with:
    uvicorn api.main:app --reload       (from the repo root)

Scoring is slow by nature: every chunk makes one local LLM call at roughly 5s, so a
five-chunk filing takes around 25s. The model, embedder and vector store are loaded
once at startup, not per request.
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clean import chunk_text, clean_text  # noqa: E402
from extract import DEFAULT_EXTRACTION, PROMPT_TEMPLATE, OllamaClient, parse_extraction  # noqa: E402
from retrieve import get_collection, get_model  # noqa: E402

MODEL_VERSION = "0.1.0"
MODEL_PATH = REPO_ROOT / "models" / "lgbm_model.pkl"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_NEIGHBOURS = 5
# Deduplicating to one hit per company collapses the pool hard - near-duplicate 8-K
# language means the closest 50 chunks often come from only a handful of issuers - so
# the pool is far wider than the 50 used at training time to still reach 5 companies.
CANDIDATE_POOL = 300
DECISION_THRESHOLD = 0.5
MAX_CHUNKS = 40

# Measured on the 2024 out-of-time holdout. Surfaced through /health because a
# consumer of risk_score needs to know the model barely separates the classes.
HOLDOUT_METRICS = {
    "split": "train 2021-2023, test 2024 (out-of-time)",
    "auroc": 0.5187,
    "auprc": 0.1347,
    "auprc_baseline": 0.1158,
    "note": "AUROC ~0.52 is close to chance; treat risk_score as weak evidence only.",
}

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model, embedder and vector store once, before serving traffic."""
    state["model"] = joblib.load(MODEL_PATH)
    state["feature_names"] = list(state["model"].feature_name_)
    state["embedder"] = get_model()
    state["collection"] = get_collection()
    state["ollama"] = OllamaClient(
        host="http://localhost:11434", model="phi3:mini", num_ctx=1536, num_predict=250
    )
    state["loaded_at"] = datetime.now(timezone.utc).isoformat()
    yield
    state.clear()


app = FastAPI(
    title="Fintech Risk Intelligence API",
    description="Scores raw 8-K filing text for the risk of a short-horizon drop.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw 8-K filing text.")


class SimilarFiling(BaseModel):
    company: str
    filing_date: str
    risk_category: str
    severity: int


class ScoreResponse(BaseModel):
    risk_score: float
    predicted_drop: bool
    dominant_risk_category: str
    avg_severity: float
    top_3_similar_filings: list[SimilarFiling]
    warning: str | None = None


def extract_chunks(chunks: list[str]) -> tuple[list[dict], int]:
    """Run each chunk through the local Phi-3 model; return fields and error count."""
    client = state["ollama"]
    results, parse_errors = [], 0

    for chunk in chunks:
        try:
            raw = client.generate(PROMPT_TEMPLATE.format(chunk_text=chunk))
            fields, error = parse_extraction(raw)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Local Ollama model unreachable or failing: {exc}",
            ) from exc
        if error:
            parse_errors += 1
        results.append(fields)

    return results, parse_errors


def find_neighbours(first_chunk: str) -> list[dict]:
    """Nearest historical chunks to the filing's first chunk, one per company.

    Training retrieved neighbours from each filing's *first* chunk, so this does the
    same. The issuer of an incoming filing is unknown, so the same-company exclusion
    used at training time cannot be applied; deduplicating to one hit per company
    keeps the neighbour set genuinely cross-company instead.
    """
    vector = state["embedder"].encode([first_chunk], normalize_embeddings=True)[0]
    collection = state["collection"]

    result = collection.query(
        query_embeddings=[vector.tolist()],
        n_results=min(CANDIDATE_POOL, collection.count()),
        include=["metadatas"],
    )

    seen, neighbours = set(), []
    for meta in result["metadatas"][0]:
        if meta["company"] in seen:
            continue
        seen.add(meta["company"])
        neighbours.append(meta)
        if len(neighbours) == TOP_K_NEIGHBOURS:
            break
    return neighbours


def build_feature_row(extractions: list[dict], parse_errors: int, neighbours: list[dict]) -> pd.DataFrame:
    """Aggregate chunk extractions to one filing-level row matching the trained model.

    Mirrors features.py. Columns are reindexed onto the model's own feature_name_ so
    a category never seen at training time cannot shift the column order.
    """
    severities = [e["severity"] for e in extractions]
    categories = pd.Series([e["risk_category"] for e in extractions])
    dominant = categories.mode().iloc[0]

    row = {
        "avg_severity": sum(severities) / len(severities),
        "negative_sentiment_ratio": sum(1 for e in extractions if e["sentiment"] == "negative") / len(extractions),
        "forward_looking_ratio": sum(1 for e in extractions if e["contains_forward_looking_statement"]) / len(extractions),
        "chunk_count": len(extractions),
        "parse_error_count": parse_errors,
        "similar_count": len(neighbours),
    }

    if neighbours:
        n_sev = [m["severity"] for m in neighbours]
        row["avg_severity_neighbours"] = sum(n_sev) / len(n_sev)
        row["negative_sentiment_ratio_neighbours"] = sum(
            1 for m in neighbours if m["sentiment"] == "negative"
        ) / len(neighbours)
        neighbour_dominant = pd.Series([m["risk_category"] for m in neighbours]).mode().iloc[0]
    else:
        # Matches training, where filings with no valid neighbours carried NaN
        # aggregates and the "missing" one-hot level.
        row["avg_severity_neighbours"] = float("nan")
        row["negative_sentiment_ratio_neighbours"] = float("nan")
        neighbour_dominant = "missing"

    row[f"dominant_risk_category_{dominant}"] = 1
    row[f"dominant_risk_category_neighbours_{neighbour_dominant}"] = 1

    frame = pd.DataFrame([row]).reindex(columns=state["feature_names"])
    # Unset one-hot levels are genuine zeros; the numeric NaNs above are left as NaN
    # because LightGBM treats them as missing by design.
    one_hot = [c for c in state["feature_names"] if c.startswith("dominant_risk_category")]
    frame[one_hot] = frame[one_hot].fillna(0)
    return frame


def build_warning(score: float, extractions: list[dict], parse_errors: int, neighbours: list[dict]) -> str | None:
    """Flag per-request reasons the score should be treated with extra caution."""
    reasons = []

    if abs(score - DECISION_THRESHOLD) < 0.10:
        reasons.append(
            f"risk_score {score:.3f} sits close to the {DECISION_THRESHOLD} decision "
            "threshold, so predicted_drop could flip either way"
        )
    if len(extractions) < 2:
        reasons.append(
            f"only {len(extractions)} chunk(s) of text, so the filing-level averages "
            "rest on very little content"
        )
    if parse_errors:
        reasons.append(
            f"{parse_errors} of {len(extractions)} chunk(s) returned malformed output "
            "from the extraction model and fell back to defaults"
        )
    if len(neighbours) < TOP_K_NEIGHBOURS:
        reasons.append(
            f"only {len(neighbours)} cross-company neighbour(s) found, so the "
            "retrieval features are weaker than at training time"
        )

    if not reasons:
        return None
    return (
        "Low confidence: " + "; ".join(reasons) + ". "
        f"Note the model's holdout AUROC is {HOLDOUT_METRICS['auroc']:.3f}, near chance."
    )


@app.get("/health")
def health() -> dict:
    """Liveness check for the service.

    Returns `status: ok` once the LightGBM model, sentence embedder and ChromaDB
    collection have loaded, along with the model version and the out-of-time holdout
    metrics, so a caller can see how much weight the scores actually deserve.
    """
    ready = "model" in state
    return {
        "status": "ok" if ready else "loading",
        "model_version": MODEL_VERSION,
        "model_file": MODEL_PATH.name,
        "loaded_at": state.get("loaded_at"),
        "holdout_metrics": HOLDOUT_METRICS,
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    """Score raw 8-K filing text for the risk of a short-horizon relative drop.

    Runs the full pipeline on the submitted text: strips SEC cover-page and signature
    boilerplate, splits it into ~500-token chunks, extracts risk fields from each
    chunk with the local Phi-3 Mini model, embeds the text, pulls the most similar
    historical filings from ChromaDB (one per company), aggregates everything to a
    single filing-level row, and runs the trained LightGBM classifier over it.

    `risk_score` is the model's raw probability. It is not calibrated - the model was
    fitted with scale_pos_weight=10 - so it ranks filings rather than stating a true
    likelihood. `warning` is populated when the request itself gives extra reason for
    caution (thin text, failed extractions, few neighbours, a borderline score).

    Expect roughly 5s per chunk, since each one makes a local LLM call.
    """
    started = time.monotonic()

    cleaned = clean_text(request.text)
    if not cleaned.strip():
        raise HTTPException(status_code=422, detail="No usable text left after cleaning.")

    chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=422, detail="Text produced no chunks.")
    if len(chunks) > MAX_CHUNKS:
        raise HTTPException(
            status_code=413,
            detail=f"Filing produced {len(chunks)} chunks, above the {MAX_CHUNKS} limit "
                   f"(~{len(chunks) * 5}s of local LLM time).",
        )

    extractions, parse_errors = extract_chunks(chunks)
    neighbours = find_neighbours(chunks[0])
    features = build_feature_row(extractions, parse_errors, neighbours)

    risk_score = float(state["model"].predict_proba(features)[0, 1])

    severities = [e["severity"] for e in extractions]
    dominant = pd.Series([e["risk_category"] for e in extractions]).mode().iloc[0]

    return ScoreResponse(
        risk_score=round(risk_score, 6),
        predicted_drop=bool(risk_score >= DECISION_THRESHOLD),
        dominant_risk_category=dominant,
        avg_severity=round(sum(severities) / len(severities), 4),
        top_3_similar_filings=[
            SimilarFiling(
                company=m["company"],
                filing_date=m["filing_date"],
                risk_category=m["risk_category"],
                severity=int(m["severity"]),
            )
            for m in neighbours[:3]
        ],
        warning=build_warning(risk_score, extractions, parse_errors, neighbours),
    )
