"""
Stratified error sampling for manual review.

The point of this project's error analysis is reading real examples and
naming failure modes, not another aggregate number. This module only does
the mechanical part -- picking a representative, stratified sample of
mistakes -- so the actual reading and clustering happens in
notebooks/03_error_analysis.ipynb where it belongs.

Three strata, matching the spec:
- high-confidence false positives: predicted very toxic, actually not
- high-confidence false negatives: predicted very non-toxic, actually toxic
- near-threshold: prediction landed close to the decision boundary either way

Also includes a label-noise flag helper: comments where a strong model
disagrees with the annotator majority are exactly the candidates for "this
isn't a model error, it's an annotation disagreement" -- being able to
quantify that fraction (rather than counting every disagreement as a bug)
is what the README's error-analysis section leans on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_error_sample(
    df: pd.DataFrame,
    label_col: str = "target",
    pred_col: str = "prediction",
    label_threshold: float = 0.5,
    n_per_stratum: int = 70,
    near_threshold_band: float = 0.1,
    high_confidence_band: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Returns a DataFrame subset of `df` with an added `error_stratum`
    column, sampling up to `n_per_stratum` rows from each of:
      - 'high_confidence_fp': label=0, pred >= 1 - high_confidence_band
      - 'high_confidence_fn': label=1, pred <= high_confidence_band
      - 'near_threshold':    |pred - 0.5| <= near_threshold_band, regardless
                             of whether the call was right or wrong -- these
                             are the cases where the model itself is unsure,
                             which is its own interesting category
    Rows are shuffled with `seed` before sampling so repeated small runs
    don't always pull the same handful of examples.
    """
    labels_bin = (df[label_col] >= label_threshold).astype(int)
    pred = df[pred_col]

    is_fp = (labels_bin == 0) & (pred >= 1 - high_confidence_band)
    is_fn = (labels_bin == 1) & (pred <= high_confidence_band)
    is_near = (pred - 0.5).abs() <= near_threshold_band

    rng = np.random.default_rng(seed)
    parts = []
    for name, mask in [
        ("high_confidence_fp", is_fp),
        ("high_confidence_fn", is_fn),
        ("near_threshold", is_near),
    ]:
        pool = df.loc[mask].copy()
        if len(pool) == 0:
            continue
        n = min(n_per_stratum, len(pool))
        sampled = pool.sample(n=n, random_state=int(rng.integers(0, 1_000_000)))
        sampled["error_stratum"] = name
        parts.append(sampled)

    if not parts:
        return df.iloc[0:0].assign(error_stratum=pd.Series(dtype="object"))
    return pd.concat(parts).reset_index(drop=True)


def flag_likely_label_noise(
    df: pd.DataFrame,
    label_col: str = "target",
    pred_col: str = "prediction",
    label_threshold: float = 0.5,
    disagreement_margin: float = 0.5,
) -> pd.Series:
    """Heuristic flag for 'this looks like an annotation disagreement, not
    a model error': the model's score and the soft label are both far from
    the decision boundary, but on opposite sides -- i.e. the model is
    *confident* in a direction the raters were not unanimous about. This
    doesn't prove annotator error, but it's the honest way to separate
    "the model is confused" from "the model disagrees with a split jury";
    read the flagged rows manually and report the fraction that hold up.

    `disagreement_margin`: how far the soft target itself sits from 0.5 in
    the *opposite* direction of a confident model call. E.g. target=0.4
    (raters split, leans non-toxic) with pred=0.95 (model very confident
    toxic) -- the raters were not unanimous, unlike a clean target=0.05.
    """
    labels_bin = (df[label_col] >= label_threshold).astype(int)
    pred = df[pred_col]
    model_confident_toxic = pred >= 0.9
    model_confident_nontoxic = pred <= 0.1
    raters_split_toward_nontoxic = (df[label_col] < label_threshold) & (df[label_col] >= label_threshold - disagreement_margin)
    raters_split_toward_toxic = (df[label_col] >= label_threshold) & (df[label_col] <= label_threshold + disagreement_margin)

    return (model_confident_toxic & raters_split_toward_nontoxic) | (
        model_confident_nontoxic & raters_split_toward_toxic
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 5000
    target = rng.random(n)
    # simulate a decent-but-imperfect model: correlated with target + noise
    prediction = np.clip(target + rng.normal(0, 0.2, n), 0, 1)
    df = pd.DataFrame({"comment_text": [f"comment {i}" for i in range(n)], "target": target, "prediction": prediction})

    sample = stratified_error_sample(df, n_per_stratum=20)
    print(sample["error_stratum"].value_counts())
    print(f"\ntotal sampled: {len(sample)}")

    noise_flag = flag_likely_label_noise(df)
    print(f"\nlikely-label-noise flagged: {noise_flag.sum()} / {n} ({noise_flag.mean():.1%})")
