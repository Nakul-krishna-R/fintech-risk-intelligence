"""Extract structured risk signals from filing chunks using a local Phi-3 Mini model.

For every chunk in data/cleaned/chunks.csv, sends the chunk text to a local Ollama
server running phi3:mini and asks it to return a JSON object with:

    risk_category                      regulatory | financial | operational |
                                        legal | leadership | none
    sentiment                          positive | negative | neutral
    severity                           integer 1-5
    contains_forward_looking_statement true / false
    key_entities                       company names, regulators, or amounts mentioned

Results are appended incrementally to data/cleaned/extractions.csv (one row per
chunk) as they come back, so a run that's interrupted after hours of work can be
resumed with the already-processed chunks skipped.

Setup (one-time):
    winget install Ollama.Ollama    (or https://ollama.com/download)
    ollama pull phi3:mini
    Ollama runs its API on http://localhost:11434 automatically once installed.

Usage:
    python src/extract.py
    python src/extract.py --limit 20          # quick test run
    python src/extract.py --no-resume          # ignore existing output, start over
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OLLAMA_HOST = "http://localhost:11434"
MODEL = "phi3:mini"

RISK_CATEGORIES = {"regulatory", "financial", "operational", "legal", "leadership", "none"}
SENTIMENTS = {"positive", "negative", "neutral"}

OUT_FIELDS = [
    "chunk_id", "company", "cik", "filing_date",
    "risk_category", "sentiment", "severity",
    "contains_forward_looking_statement", "key_entities",
    "parse_error",
]

PROMPT_TEMPLATE = """You are a financial risk analyst extracting structured data from an excerpt of an SEC 8-K filing.

Return ONLY a single JSON object with exactly these fields, no other text:
- "risk_category": one of "regulatory", "financial", "operational", "legal", "leadership", "none"
- "sentiment": one of "positive", "negative", "neutral"
- "severity": integer from 1 to 5 (1 = minimal/no risk, 5 = severe risk)
- "contains_forward_looking_statement": true or false
- "key_entities": a list of strings naming companies, regulators, or monetary amounts mentioned

Filing excerpt:
\"\"\"
{chunk_text}
\"\"\"

JSON:"""

# The model's own text-fence habits, stripped before JSON parsing in case "format":
# "json" mode doesn't fully suppress them.
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

DEFAULT_EXTRACTION = {
    "risk_category": "none",
    "sentiment": "neutral",
    "severity": 1,
    "contains_forward_looking_statement": False,
    "key_entities": [],
}


class OllamaClient:
    def __init__(self, host: str, model: str, num_ctx: int, num_predict: int, retries: int = 3):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.retries = retries

    def check_available(self) -> None:
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=5).read()
        except Exception as exc:
            sys.exit(
                f"Can't reach Ollama at {self.host} ({exc}).\n"
                "Make sure the Ollama app/service is running (it starts automatically "
                "after installing on Windows), then retry."
            )

    def generate(self, prompt: str) -> str:
        """Call /api/generate and return the raw response text, retrying on errors."""
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_ctx": self.num_ctx,
                    "num_predict": self.num_predict,
                },
            }
        ).encode("utf-8")

        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(
                    f"{self.host}/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180) as resp:
                    body = json.loads(resp.read())
                return body.get("response", "")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 * attempt)
        raise RuntimeError(f"Ollama request failed after {self.retries} attempts: {last_error}")


def parse_extraction(raw_response: str) -> tuple[dict, str | None]:
    """Parse and validate the model's JSON response.

    Returns (fields, error). On any failure `fields` is a safe default dict and
    `error` describes what went wrong, so a bad response never crashes the run or
    drops the chunk.
    """
    match = JSON_OBJECT_RE.search(raw_response)
    if not match:
        return dict(DEFAULT_EXTRACTION), "no JSON object found in response"

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return dict(DEFAULT_EXTRACTION), f"invalid JSON: {exc}"

    if not isinstance(data, dict):
        return dict(DEFAULT_EXTRACTION), "response was not a JSON object"

    errors = []
    result = dict(DEFAULT_EXTRACTION)

    category = str(data.get("risk_category", "")).strip().lower()
    if category in RISK_CATEGORIES:
        result["risk_category"] = category
    else:
        errors.append(f"bad risk_category {category!r}")

    sentiment = str(data.get("sentiment", "")).strip().lower()
    if sentiment in SENTIMENTS:
        result["sentiment"] = sentiment
    else:
        errors.append(f"bad sentiment {sentiment!r}")

    try:
        severity = int(data.get("severity"))
        result["severity"] = max(1, min(5, severity))
    except (TypeError, ValueError):
        errors.append(f"bad severity {data.get('severity')!r}")

    forward_looking = data.get("contains_forward_looking_statement")
    if isinstance(forward_looking, bool):
        result["contains_forward_looking_statement"] = forward_looking
    elif isinstance(forward_looking, str) and forward_looking.strip().lower() in ("true", "false"):
        result["contains_forward_looking_statement"] = forward_looking.strip().lower() == "true"
    else:
        errors.append(f"bad contains_forward_looking_statement {forward_looking!r}")

    entities = data.get("key_entities")
    if isinstance(entities, list):
        result["key_entities"] = [str(e).strip() for e in entities if str(e).strip()]
    elif entities:
        errors.append(f"bad key_entities {entities!r}")

    return result, "; ".join(errors) if errors else None


def extract_chunk(client: OllamaClient, chunk_text: str) -> tuple[dict, str | None]:
    prompt = PROMPT_TEMPLATE.format(chunk_text=chunk_text)
    try:
        raw = client.generate(prompt)
    except RuntimeError as exc:
        return dict(DEFAULT_EXTRACTION), str(exc)
    return parse_extraction(raw)


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with out_path.open(newline="", encoding="utf-8") as f:
        return {row["chunk_id"] for row in csv.DictReader(f)}


def process(args: argparse.Namespace) -> int:
    chunks_path = Path(args.chunks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(args.host, args.model, args.num_ctx, args.num_predict)
    client.check_available()

    with chunks_path.open(newline="", encoding="utf-8") as f:
        chunks = list(csv.DictReader(f))
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    done = set() if args.no_resume else load_done(out_path)
    if done:
        print(f"Resuming: {len(done)} chunks already extracted")

    write_header = args.no_resume or not out_path.exists()
    mode = "w" if args.no_resume else "a"
    parse_errors = 0
    processed = 0
    t0 = time.time()

    with out_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if write_header:
            writer.writeheader()

        for i, row in enumerate(chunks, start=1):
            if row["chunk_id"] in done:
                continue

            fields, error = extract_chunk(client, row["chunk_text"])
            if error:
                parse_errors += 1

            writer.writerow(
                {
                    "chunk_id": row["chunk_id"],
                    "company": row["company"],
                    "cik": row["cik"],
                    "filing_date": row["filing_date"],
                    "risk_category": fields["risk_category"],
                    "sentiment": fields["sentiment"],
                    "severity": fields["severity"],
                    "contains_forward_looking_statement": fields["contains_forward_looking_statement"],
                    "key_entities": "; ".join(fields["key_entities"]),
                    "parse_error": error or "",
                }
            )
            f.flush()
            processed += 1

            if processed % 100 == 0 or i == len(chunks):
                elapsed = time.time() - t0
                rate = elapsed / processed
                remaining = len(chunks) - i
                eta_min = remaining * rate / 60
                print(
                    f"[{i}/{len(chunks)}] processed={processed} parse_errors={parse_errors} "
                    f"avg={rate:.1f}s/chunk eta={eta_min:.0f}min"
                )

    print(f"\nDone: {processed} chunks extracted -> {out_path}")
    if parse_errors:
        print(f"  {parse_errors} chunk(s) had malformed/unexpected model output "
              f"(defaults were recorded; see the parse_error column)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--chunks",
        default=REPO_ROOT / "data" / "cleaned" / "chunks.csv",
        help="input chunks CSV (default: data/cleaned/chunks.csv)",
    )
    parser.add_argument(
        "--out",
        default=REPO_ROOT / "data" / "cleaned" / "extractions.csv",
        help="output CSV path (default: data/cleaned/extractions.csv)",
    )
    parser.add_argument("--model", default=MODEL, help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--host", default=OLLAMA_HOST, help=f"Ollama API host (default: {OLLAMA_HOST})")
    parser.add_argument("--num-ctx", type=int, default=1536, help="model context window (default: 1536)")
    parser.add_argument("--num-predict", type=int, default=250, help="max output tokens (default: 250)")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N chunks (testing)")
    parser.add_argument("--no-resume", action="store_true", help="ignore existing output and start over")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(process(parse_args()))
