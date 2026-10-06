"""Generate the three summary charts for the risk pipeline.

    models/risk_category_distribution.png   risk categories, split by sentiment
    models/severity_by_risk_category.png    mean severity per risk category
    models/precision_recall_curve.png       2024 holdout PR curve vs. baseline

Chart 3 rebuilds the holdout matrix through train.py's own load_split/build_matrices
so the columns fed to the model here are identical to the ones it was fitted on -
in particular the leakage columns stay excluded.

Usage:
    python src/visualise.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Palette: validated with the dataviz skill's validate_palette.js against the light
# surface (#fcfcfb). Sentiment is polarity, so the two poles are warm/cool opposites
# with the neutral case between them; all three clear the lightness band, chroma
# floor, CVD separation and normal-vision floor on the adjacent pairlist.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_BLUE = "#2a78d6"
SENTIMENT_COLORS = {
    "negative": "#e34948",
    "neutral": "#2a78d6",
    "positive": "#1baf7a",
}
# Stacked left-to-right in this order so adjacent pairs match what was validated.
SENTIMENT_ORDER = ["negative", "neutral", "positive"]


def style_axes(ax, xlabel: str = "", ylabel: str = "") -> None:
    """Recessive chrome: hairline solid grid, no box, muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)


def new_figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved -> {path}")


def chart_risk_distribution(extractions: pd.DataFrame, out_path: Path) -> None:
    """Horizontal stacked bar: chunks per risk category, split by sentiment."""
    pivot = (
        extractions.pivot_table(
            index="risk_category", columns="sentiment", aggfunc="size", fill_value=0
        )
        .reindex(columns=SENTIMENT_ORDER, fill_value=0)
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values().index]  # largest at top

    fig, ax = new_figure(9.5, 5.2)
    left = np.zeros(len(pivot))

    for sentiment in SENTIMENT_ORDER:
        values = pivot[sentiment].to_numpy()
        ax.barh(
            pivot.index,
            values,
            left=left,
            height=0.62,
            color=SENTIMENT_COLORS[sentiment],
            label=sentiment,
            # Surface-coloured edge produces the 2px gap between fills rather than
            # drawing a contrasting border around each segment.
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        # Label only segments with room for the text, so nothing gets clipped.
        for y, (value, offset) in enumerate(zip(values, left)):
            if value >= pivot.to_numpy().sum() * 0.025:
                ax.text(
                    offset + value / 2, y, f"{value:,}",
                    ha="center", va="center", color="#ffffff", fontsize=9,
                )
        left += values

    totals = pivot.sum(axis=1).to_numpy()
    for y, total in enumerate(totals):
        ax.text(total + totals.max() * 0.012, y, f"{total:,}",
                ha="left", va="center", color=INK_SECONDARY, fontsize=9.5)

    ax.set_xlim(0, totals.max() * 1.1)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax, xlabel="Chunks")
    ax.set_title(
        f"Risk category distribution by sentiment  ({len(extractions):,} chunks)",
        color=INK_PRIMARY, fontsize=13, pad=14, loc="left",
    )
    legend = ax.legend(
        title="Sentiment", frameon=False, loc="lower right",
        fontsize=9.5, title_fontsize=9.5,
    )
    plt.setp(legend.get_texts(), color=INK_SECONDARY)
    plt.setp(legend.get_title(), color=INK_SECONDARY)
    save(fig, out_path)


def chart_severity_by_category(extractions: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar: mean severity per risk category.

    Risk categories are nominal, so every bar takes the same hue - shading by value
    would double-encode the bar length and burn the colour channel for nothing.
    """
    grouped = extractions.groupby("risk_category")["severity"].agg(["mean", "count"])
    grouped = grouped.sort_values("mean")

    fig, ax = new_figure(9.5, 5.2)
    ax.barh(grouped.index, grouped["mean"], height=0.62, color=SERIES_BLUE)

    for y, (mean, count) in enumerate(zip(grouped["mean"], grouped["count"])):
        ax.text(mean + 0.045, y, f"{mean:.2f}", ha="left", va="center",
                color=INK_PRIMARY, fontsize=9.5)
        ax.text(0.06, y, f"n={count:,}", ha="left", va="center",
                color="#ffffff", fontsize=8.5)

    ax.set_xlim(0, max(5.0, grouped["mean"].max() * 1.18))
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax, xlabel="Mean severity  (1 = minimal, 5 = severe)")
    ax.set_title("Average severity by risk category", color=INK_PRIMARY,
                 fontsize=13, pad=14, loc="left")
    save(fig, out_path)


def chart_precision_recall(features_path: Path, model_path: Path, out_path: Path,
                           train_end: str, test_start: str) -> None:
    """PR curve on the 2024 holdout, against the random-classifier baseline."""
    from train import build_matrices, load_split

    # Those helpers narrate to stdout; this script has its own progress output.
    with contextlib.redirect_stdout(io.StringIO()):
        train_df, test_df = load_split(features_path, train_end, test_start)
        _, X_test, _, y_test = build_matrices(train_df, test_df)

    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    baseline = float(y_test.mean())

    fig, ax = new_figure(8.2, 5.6)
    ax.plot(recall, precision, color=SERIES_BLUE, linewidth=2,
            label=f"LightGBM  (AUPRC = {auprc:.3f})")
    ax.axhline(baseline, color=INK_MUTED, linewidth=1.5, linestyle="--",
               label=f"Random baseline  ({baseline:.3f})")

    # sklearn's curve ends at (recall 0, precision 1); at very low recall precision is
    # decided by one or two filings and spikes to 1.0. Scaling to that leaves the real
    # curve squashed into the bottom of the plot, so the axis is set from the region
    # where recall is meaningful and the clipping is stated on the chart.
    settled = precision[recall >= 0.05]
    y_max = max(0.40, float(settled.max()) * 1.15) if settled.size else 1.0
    clipped = float(precision.max()) > y_max

    ax.set_xlim(0, 1)
    ax.set_ylim(0, y_max)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    style_axes(ax, xlabel="Recall", ylabel="Precision")
    ax.set_title(
        f"Precision-recall, 2024 holdout  ({len(y_test):,} filings, "
        f"{int(y_test.sum())} positive)",
        color=INK_PRIMARY, fontsize=13, pad=14, loc="left",
    )
    legend = ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    plt.setp(legend.get_texts(), color=INK_SECONDARY)

    # The headline comparison is curve vs. baseline, so state the ratio outright.
    note = f"lift over baseline: {auprc / baseline:.2f}x"
    if clipped:
        note += f"      y-axis clipped at {y_max:.2f}; precision reaches " \
                f"{precision.max():.2f} below 0.05 recall (a handful of filings)"
    ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", va="bottom",
            color=INK_MUTED, fontsize=8.5)
    save(fig, out_path)


def main(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    extractions = pd.read_csv(args.extractions)
    extractions["severity"] = pd.to_numeric(extractions["severity"], errors="coerce")

    print(f"Loaded {len(extractions):,} chunks from {args.extractions}")

    print("\nChart 1: risk category distribution by sentiment")
    chart_risk_distribution(extractions, out_dir / "risk_category_distribution.png")

    print("\nChart 2: average severity by risk category")
    chart_severity_by_category(extractions, out_dir / "severity_by_risk_category.png")

    print("\nChart 3: precision-recall curve (2024 holdout)")
    chart_precision_recall(
        Path(args.features), Path(args.model),
        out_dir / "precision_recall_curve.png",
        args.train_end, args.test_start,
    )

    print("\nDone: 3 charts written to", out_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--extractions",
        default=REPO_ROOT / "data" / "cleaned" / "extractions.csv",
        help="chunk-level extractions (default: data/cleaned/extractions.csv)",
    )
    parser.add_argument(
        "--features",
        default=REPO_ROOT / "data" / "features" / "feature_matrix.csv",
        help="feature matrix (default: data/features/feature_matrix.csv)",
    )
    parser.add_argument(
        "--model",
        default=REPO_ROOT / "models" / "lgbm_model.pkl",
        help="trained model (default: models/lgbm_model.pkl)",
    )
    parser.add_argument("--out-dir", default=REPO_ROOT / "models", help="output dir (default: models/)")
    parser.add_argument("--train-end", default="2024-01-01", help="train/test boundary used at fit time")
    parser.add_argument("--test-start", default="2024-01-01", help="holdout start date")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
