"""
Stage 0 baseline: TF-IDF (word + char n-grams) -> Logistic Regression.

Not filler. This trains in minutes on CPU and gets a visibly-decent
overall AUC with a visibly-bad bias score, which is what makes the later
"the transformer bought me +X overall AUC but +Y BPSN" sentence in the
README meaningful instead of a single unanchored number.

Runnable standalone once data/raw/train.csv exists:
    python src/baseline.py
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data import load_raw, stratified_subsample, train_val_split
from metrics import IDENTITY_COLUMNS, compute_bias_metrics

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path = ROOT / "config" / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_vectorizers(cfg: dict) -> tuple[Pipeline, Pipeline]:
    """HashingVectorizer -> TfidfTransformer, not plain TfidfVectorizer.

    A vocabulary-based CountVectorizer/TfidfVectorizer has to build an
    explicit {n-gram: index} dict across the WHOLE corpus before it can
    prune down to `max_features` -- for char_wb (2,5) over 1.7M comments
    (some up to ~1000 chars, see EDA) that intermediate vocabulary blows
    past available RAM before pruning ever happens (hit this directly:
    MemoryError inside sklearn's `_count_vocab`, ~40 minutes into a fit).
    HashingVectorizer sidesteps this entirely -- it hashes each n-gram
    straight into a fixed-size feature space with no vocabulary dict at
    all, so memory is bounded by `n_features` and the corpus's total
    non-zero count, not by how many distinct n-grams exist. `norm=None` +
    a separate `TfidfTransformer` keeps the actual TF-IDF weighting (IDF
    needs document frequencies, which HashingVectorizer alone can't give
    you -- it has no notion of "this hash bucket = this term").
    """
    word_vec = Pipeline([
        ("hash", HashingVectorizer(
            analyzer="word",
            ngram_range=tuple(cfg["baseline_tfidf"]["word_ngram_range"]),
            n_features=2**20,
            alternate_sign=False,
            norm=None,
        )),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
    ])
    char_vec = Pipeline([
        ("hash", HashingVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(cfg["baseline_tfidf"]["char_ngram_range"]),
            n_features=2**20,
            alternate_sign=False,
            norm=None,
        )),
        ("tfidf", TfidfTransformer(sublinear_tf=True)),
    ])
    return word_vec, char_vec


def main(args: argparse.Namespace) -> None:
    cfg = load_config()
    raw_path = Path(args.data) if args.data else ROOT / cfg["paths"]["raw_train_csv"]

    usecols = ["comment_text", "target"] + IDENTITY_COLUMNS
    print(f"Loading {raw_path} ...")
    df = load_raw(raw_path, usecols=usecols)
    df["comment_text"] = df["comment_text"].fillna("")

    if args.sample:
        # Plain random subsample -- for quick, throwaway smoke runs only.
        # Doesn't guarantee subgroup coverage, so its bias table can be noisy
        # or NaN for rare subgroups.
        df = df.sample(n=min(args.sample, len(df)), random_state=cfg["seed"]).reset_index(drop=True)
        print(f"Subsampled to {len(df)} rows for a quick run (--sample {args.sample}).")
    elif not args.full:
        # Default: a stratified subsample (see src/data.py) rather than the
        # full 1.8M rows. This is a disclosed, deliberate tradeoff for this
        # dev machine's 8GB RAM -- HashingVectorizer avoids building an
        # explicit vocabulary, but the final char-n-gram sparse matrix for
        # the FULL corpus still needs to hold ~1 billion (doc, feature)
        # instances in memory before it's compressed to CSR, which doesn't
        # fit here regardless of vectorization strategy. Unlike a plain
        # random subsample, stratified_subsample tops up every one of the
        # 9 scored subgroups to a floor count, so the bias table stays
        # meaningful at this size instead of going noisy/NaN on rare
        # subgroups. Pass --full on a machine with more RAM (or on Kaggle)
        # to use every row.
        df = stratified_subsample(
            df,
            n=cfg["data"]["dev_subsample_size"],
            seed=cfg["seed"],
            min_per_subgroup=cfg["data"]["min_per_subgroup_dev"],
        )
        print(f"Stratified-subsampled to {len(df)} rows (RAM-constrained default; pass --full to use all rows).")

    train_df, val_df = train_val_split(df, val_size=cfg["data"]["val_size"], seed=cfg["seed"])
    print(f"train={len(train_df)} val={len(val_df)}")

    word_vec, char_vec = build_vectorizers(cfg)
    print("Fitting TF-IDF vectorizers...")
    X_train_word = word_vec.fit_transform(train_df["comment_text"])
    X_train_char = char_vec.fit_transform(train_df["comment_text"])
    X_train = hstack([X_train_word, X_train_char]).tocsr()

    X_val_word = word_vec.transform(val_df["comment_text"])
    X_val_char = char_vec.transform(val_df["comment_text"])
    X_val = hstack([X_val_word, X_val_char]).tocsr()

    y_train = (train_df["target"] >= cfg["data"]["toxic_threshold"]).astype(int)

    print("Training LogisticRegression...")
    clf = LogisticRegression(
        C=cfg["baseline_tfidf"]["C"],
        solver="liblinear",
        max_iter=1000,
        class_weight=None,  # deliberately unweighted here; the weighting
        # ablation belongs to the transformer stage per the README table.
    )
    clf.fit(X_train, y_train)

    val_scores = clf.predict_proba(X_val)[:, 1]
    val_df = val_df.copy()
    val_df["prediction"] = val_scores

    result = compute_bias_metrics(val_df, label_col="target", pred_col="prediction")
    print("\n=== Baseline TF-IDF + LogisticRegression: bias metric report ===")
    print(result.per_subgroup)
    print()
    print(result.summary())

    out_dir = ROOT / cfg["paths"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_subgroup.to_csv(out_dir / "baseline_tfidf_per_subgroup.csv")
    result.summary().to_csv(out_dir / "baseline_tfidf_summary.csv")

    # Full per-row val predictions -- consumed by notebooks/03_error_analysis.ipynb
    # (src/error_analysis.py's stratified sampler) and by the threshold sweep.
    val_out_cols = ["comment_text", "target", "prediction"] + IDENTITY_COLUMNS
    val_df[val_out_cols].to_csv(out_dir / "baseline_tfidf_val_predictions.csv", index=False)
    print(f"\nSaved bias tables + val predictions to {out_dir}")

    if args.save_model:
        model_dir = ROOT / cfg["paths"]["model_dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "baseline_tfidf.pkl", "wb") as f:
            pickle.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf}, f)
        print(f"Saved model to {model_dir / 'baseline_tfidf.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=None, help="path to train.csv (default: config paths.raw_train_csv)")
    parser.add_argument("--sample", type=int, default=None, help="plain random subsample this many rows (quick smoke run only)")
    parser.add_argument("--full", action="store_true", help="use every row (needs a fair amount of RAM -- see build_vectorizers docstring)")
    parser.add_argument("--save-model", action="store_true", help="pickle the fitted vectorizers + classifier")
    main(parser.parse_args())
