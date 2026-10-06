"""Train a LightGBM classifier to predict stock_drop, with out-of-time validation.

Split is strictly chronological: train on 2021-2023 filings, test on 2024 only.
A random split would leak the future into the training set, since filings from the
same company weeks apart share both language and price path.

Hyperparameters are tuned with Optuna (50 trials) maximising AUPRC over 3-fold CV on
the training set; the best configuration is then refit on the full training set and
scored once on the 2024 holdout.

Outputs:
    models/lgbm_model.pkl           fitted LGBMClassifier (joblib)
    models/feature_importance.png   gain-based feature importance

Usage:
    python src/train.py
    python src/train.py --trials 10 --no-tune        # fast sanity run
    python src/train.py --cv-strategy timeseries    # time-aware inner CV
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lightgbm as lgb
import optuna
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET = "stock_drop"
CATEGORICAL = ["dominant_risk_category", "dominant_risk_category_neighbours"]

# Identity columns: unique or near-unique per row, so the model would memorise rather
# than generalise.
ID_COLUMNS = ["accession_number", "ticker", "company", "cik", "filing_date"]

# Target leakage. stock_drop is *defined* as excess_return < threshold, and the rest
# are the prices and returns that computed it, so any of them hands the model the
# answer. They live in feature_matrix.csv only so the label can be audited.
LEAKAGE_COLUMNS = [
    "baseline_price", "end_price", "stock_return", "benchmark_return", "excess_return",
    # retained so an older matrix regenerated with --drop-method return/trough is
    # still handled correctly
    "min_price_90d", "max_drawdown_90d", "fwd_return_90d",
]


def load_split(path: Path, train_end: str, test_start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df["filing_date"] = pd.to_datetime(df["filing_date"])

    train = df[df["filing_date"] < train_end].copy()
    test = df[(df["filing_date"] >= test_start)].copy()

    print("=" * 70)
    print("DATA")
    print("=" * 70)
    print(f"  source            {path}")
    print(f"  total filings     {len(df)}")
    print(f"  train (< {train_end})   {len(train):>5} rows, "
          f"{int(train[TARGET].sum()):>4} positive ({train[TARGET].mean():.2%})")
    print(f"  test  (>= {test_start})  {len(test):>5} rows, "
          f"{int(test[TARGET].sum()):>4} positive ({test[TARGET].mean():.2%})")
    print(f"  train date range  {train['filing_date'].min().date()} -> {train['filing_date'].max().date()}")
    print(f"  test  date range  {test['filing_date'].min().date()} -> {test['filing_date'].max().date()}")
    return train, test


def build_matrices(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """One-hot the categoricals and drop identity + leakage columns."""
    dropped = ID_COLUMNS + LEAKAGE_COLUMNS + [TARGET]

    def prep(frame: pd.DataFrame) -> pd.DataFrame:
        X = frame.drop(columns=[c for c in dropped if c in frame.columns])
        # Blank neighbour category means "no valid neighbours"; keep it as its own
        # level rather than letting get_dummies silently zero the row out.
        for col in CATEGORICAL:
            X[col] = X[col].fillna("missing").astype(str)
        return pd.get_dummies(X, columns=CATEGORICAL, prefix=CATEGORICAL)

    X_train = prep(train)
    X_test = prep(test)
    # Align so a category present in only one split can't shift column order.
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"\n  features          {X_train.shape[1]}")
    print(f"  dropped (identity) {ID_COLUMNS}")
    present_leakage = [c for c in LEAKAGE_COLUMNS if c in train.columns]
    print(f"  dropped (leakage)  {present_leakage}")
    print("                     ^ stock_drop is derived from excess_return, so these")
    print("                       would give a trivially perfect model.")
    print(f"  numeric NaNs kept for LightGBM: "
          f"{int(X_train.isna().sum().sum())} train / {int(X_test.isna().sum().sum())} test")

    return X_train, X_test, train[TARGET], test[TARGET]


def make_cv(strategy: str, folds: int, seed: int):
    if strategy == "timeseries":
        return TimeSeriesSplit(n_splits=folds)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def cv_auprc(params: dict, X: pd.DataFrame, y: pd.Series, cv, seed: int) -> float:
    scores = []
    for fold_train, fold_valid in cv.split(X, y):
        model = lgb.LGBMClassifier(**params, random_state=seed, verbosity=-1)
        model.fit(X.iloc[fold_train], y.iloc[fold_train])
        probs = model.predict_proba(X.iloc[fold_valid])[:, 1]
        scores.append(average_precision_score(y.iloc[fold_valid], probs))
    return float(np.mean(scores))


def tune(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> dict:
    cv = make_cv(args.cv_strategy, args.cv_folds, args.seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "scale_pos_weight": args.scale_pos_weight,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        return cv_auprc(params, X, y, cv, args.seed)

    print("\n" + "=" * 70)
    print(f"TUNING  ({args.trials} Optuna trials, {args.cv_folds}-fold {args.cv_strategy} CV, metric=AUPRC)")
    print("=" * 70)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    completed = {"n": 0}

    def report(study_: optuna.Study, trial_: optuna.trial.FrozenTrial) -> None:
        completed["n"] += 1
        if completed["n"] % 10 == 0 or completed["n"] == args.trials:
            print(f"  trial {completed['n']:>3}/{args.trials}  "
                  f"best CV AUPRC = {study_.best_value:.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        study.optimize(objective, n_trials=args.trials, callbacks=[report])

    print(f"\n  best CV AUPRC     {study.best_value:.4f}")
    print("  best params:")
    for key, value in sorted(study.best_params.items()):
        print(f"    {key:20} {value}")
    return study.best_params


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    base_rate = y_test.mean()

    print("\n" + "=" * 70)
    print("HOLDOUT RESULTS (2024, never seen in training or tuning)")
    print("=" * 70)
    print(f"  AUROC             {auroc:.4f}   (0.5 = coin flip)")
    print(f"  AUPRC             {auprc:.4f}   (baseline = base rate = {base_rate:.4f})")
    print(f"  AUPRC lift        {auprc / base_rate:.2f}x over always-predicting-positive")

    matrix = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = matrix.ravel()
    print(f"\n  Confusion matrix @ threshold 0.50")
    print(f"                    predicted")
    print(f"                    no-drop    drop")
    print(f"    actual no-drop   {tn:>6}  {fp:>6}")
    print(f"    actual drop      {fn:>6}  {tp:>6}")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )
    print(f"\n    precision {precision:.4f}  recall {recall:.4f}  f1 {f1:.4f}")

    print("\n  Full classification report @ 0.50:")
    print("   ", classification_report(y_test, preds, target_names=["no-drop", "drop"],
                                       zero_division=0).replace("\n", "\n    "))

    # scale_pos_weight pushes probabilities up, so 0.5 is rarely the useful operating
    # point; show the best-F1 threshold as a fairer read of achievable performance.
    thresholds = np.linspace(0.05, 0.95, 91)
    f1s = [
        precision_recall_fscore_support(
            y_test, (probs >= t).astype(int), average="binary", zero_division=0
        )[2]
        for t in thresholds
    ]
    best_i = int(np.argmax(f1s))
    best_t = thresholds[best_i]
    best_preds = (probs >= best_t).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y_test, best_preds).ravel()
    p2, r2, f2, _ = precision_recall_fscore_support(
        y_test, best_preds, average="binary", zero_division=0
    )
    print(f"  Best-F1 threshold {best_t:.2f}  ->  precision {p2:.4f}  recall {r2:.4f}  f1 {f2:.4f}")
    print(f"    confusion: tn={tn2} fp={fp2} fn={fn2} tp={tp2}")


def plot_importance(model, out_path: Path, top_n: int) -> None:
    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=model.booster_.feature_name(),
    ).sort_values(ascending=False)

    top = importance.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(top))))
    ax.barh(top.index, top.values, color="#4C78A8")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"LightGBM feature importance (top {len(top)}, gain)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (gain)")
    print("=" * 70)
    for name, value in importance.head(top_n).items():
        print(f"  {name:45} {value:12.2f}")
    zero = int((importance == 0).sum())
    if zero:
        print(f"  ({zero} feature(s) contributed zero gain)")
    print(f"\n  saved -> {out_path}")


def main(args: argparse.Namespace) -> int:
    train_df, test_df = load_split(Path(args.features), args.train_end, args.test_start)
    if test_df.empty or train_df.empty:
        sys.exit("Train or test split is empty; check --train-end / --test-start.")

    X_train, X_test, y_train, y_test = build_matrices(train_df, test_df)

    if args.no_tune:
        print("\n  --no-tune: using LightGBM defaults")
        best_params = {}
    else:
        best_params = tune(X_train, y_train, args)

    print("\n" + "=" * 70)
    print("FINAL MODEL (refit on full training set)")
    print("=" * 70)
    model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=args.scale_pos_weight,
        random_state=args.seed,
        verbosity=-1,
        **best_params,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    print(f"  trained on {len(X_train)} filings, {X_train.shape[1]} features, "
          f"scale_pos_weight={args.scale_pos_weight}")

    evaluate(model, X_test, y_test)

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    plot_importance(model, models_dir / "feature_importance.png", args.top_n)

    model_path = models_dir / "lgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"  model saved -> {model_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--features",
        default=REPO_ROOT / "data" / "features" / "feature_matrix.csv",
        help="input feature matrix (default: data/features/feature_matrix.csv)",
    )
    parser.add_argument("--models-dir", default=REPO_ROOT / "models", help="output dir (default: models/)")
    parser.add_argument("--train-end", default="2024-01-01", help="train on filings before this date")
    parser.add_argument("--test-start", default="2024-01-01", help="test on filings from this date on")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials (default: 50)")
    parser.add_argument("--cv-folds", type=int, default=3, help="inner CV folds (default: 3)")
    parser.add_argument(
        "--cv-strategy",
        default="stratified",
        choices=["stratified", "timeseries"],
        help="inner CV scheme (default: stratified)",
    )
    parser.add_argument("--scale-pos-weight", type=float, default=10.0, help="class imbalance weight (default: 10.0)")
    parser.add_argument("--top-n", type=int, default=20, help="features shown in the importance plot (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--no-tune", action="store_true", help="skip Optuna, use LightGBM defaults")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
