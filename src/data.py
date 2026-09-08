"""
Data loading, sample weighting, and splitting for the Jigsaw Unintended
Bias dataset.

Sample-weighting scheme
------------------------
We do NOT use SMOTE or naive class oversampling here -- see the README for
why (short version: SMOTE is meaningless on transformer text embeddings,
and naively oversampling the toxic class amplifies whatever identity
correlations already exist in that class, making the bias problem worse,
not better).

Instead we use the metric-derived weighting scheme published with the
competition. It is derived directly from the evaluation metric (see
`metrics.py`), not from the class imbalance in isolation:

    w  = 0.25                                          # base weight
    w += 0.25 * [comment mentions any of the 9 subgroups]
    w += 0.25 * [toxic AND mentions no subgroup]         # background positive
    w += 0.25 * [non-toxic AND mentions a subgroup]      # subgroup negative

The last term is the important one: it upweights exactly the examples
whose misclassification tanks BPSN AUC (benign comments that mention an
identity). The third term does the mirror job for BNSP (real attacks that
don't happen to mention a tracked identity, which would otherwise be
drowned out once we start upweighting subgroup mentions). Weights are
un-normalized on purpose -- what matters for a weighted loss is their
*relative* scale, and BCE loss functions accept per-example weights
directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import AUX_TOXICITY_COLUMNS, IDENTITY_COLUMNS

DEFAULT_TOXIC_THRESHOLD = 0.5
DEFAULT_IDENTITY_THRESHOLD = 0.5


def load_raw(path: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the raw Jigsaw train.csv.

    `usecols=None` loads every column (comment_text, target, the 9 scored
    identities, ~15 other identity columns, the 6 aux subtypes, and some
    metadata columns like rating counts). Pass an explicit list to save
    memory during development.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download train.csv from the Kaggle "
            "'jigsaw-unintended-bias-in-toxicity-classification' "
            "competition and place it under data/raw/."
        )
    return pd.read_csv(path, usecols=usecols)


def mentions_any_subgroup(
    df: pd.DataFrame,
    identity_cols: list[str] = IDENTITY_COLUMNS,
    threshold: float = DEFAULT_IDENTITY_THRESHOLD,
) -> pd.Series:
    """Boolean: does this row have >=1 of the 9 scored identities annotated
    at >= threshold? NaN identity values (the 1.35M un-annotated rows)
    count as "not mentioned", which is the correct interpretation -- we
    have no annotator evidence either way, so we do not claim membership.
    """
    present = df[identity_cols].fillna(0.0) >= threshold
    return present.any(axis=1)


def compute_sample_weights(
    df: pd.DataFrame,
    target_col: str = "target",
    identity_cols: list[str] = IDENTITY_COLUMNS,
    toxic_threshold: float = DEFAULT_TOXIC_THRESHOLD,
    identity_threshold: float = DEFAULT_IDENTITY_THRESHOLD,
) -> pd.Series:
    """Metric-derived sample weights. See module docstring for the
    derivation. Returns a float Series aligned to df.index, range [0.25, 1.0].
    """
    toxic = df[target_col] >= toxic_threshold
    has_identity = mentions_any_subgroup(df, identity_cols, identity_threshold)

    w = pd.Series(0.25, index=df.index, dtype="float64")
    w += 0.25 * has_identity.astype("float64")
    w += 0.25 * (toxic & ~has_identity).astype("float64")
    w += 0.25 * (~toxic & has_identity).astype("float64")
    return w


def compute_inverse_frequency_weights(df: pd.DataFrame, target_col: str = "target", toxic_threshold: float = DEFAULT_TOXIC_THRESHOLD) -> pd.Series:
    """Plain class-balance baseline for the ablation table (NOT the
    scheme we ship): weight = 1 / P(class). Included only so the README's
    "no weighting vs inverse-frequency vs metric-derived" comparison has a
    fair, standard second row.
    """
    toxic = df[target_col] >= toxic_threshold
    p_toxic = toxic.mean()
    p_nontoxic = 1 - p_toxic
    w = pd.Series(np.where(toxic, 1.0 / p_toxic, 1.0 / p_nontoxic), index=df.index)
    return w


def train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.05,
    seed: int = 42,
    target_col: str = "target",
    toxic_threshold: float = DEFAULT_TOXIC_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on (toxic label x has-any-identity) so the rare
    "toxic and mentions an identity" stratum -- the one BPSN/BNSP care
    about most -- doesn't get unlucky in a plain random split.
    """
    from sklearn.model_selection import train_test_split

    strata = (
        (df[target_col] >= toxic_threshold).astype(int).astype(str)
        + "_"
        + mentions_any_subgroup(df).astype(int).astype(str)
    )
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=strata
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def stratified_subsample(
    df: pd.DataFrame,
    n: int = 200_000,
    seed: int = 42,
    target_col: str = "target",
    toxic_threshold: float = DEFAULT_TOXIC_THRESHOLD,
    min_per_subgroup: int = 500,
) -> pd.DataFrame:
    """Fast-iteration dev subsample.

    Plain proportional stratified sampling on (toxic x has-identity) would
    still leave individual sparse subgroups (e.g. `muslim`) with very few
    rows once you're down to 200k, which makes the bias metrics on the dev
    subsample noisy to the point of being useless for iteration. So: do a
    proportional stratified sample first, then top up each of the 9 scored
    subgroups to at least `min_per_subgroup` rows by adding back randomly
    sampled rows that mention it (dropping duplicates). This keeps the
    toxic base rate roughly intact while guaranteeing every subgroup table
    is estimated on a meaningful sample during development.
    """
    from sklearn.model_selection import train_test_split

    n = min(n, len(df))
    strata = (
        (df[target_col] >= toxic_threshold).astype(int).astype(str)
        + "_"
        + mentions_any_subgroup(df).astype(int).astype(str)
    )
    sample, _ = train_test_split(
        df, train_size=n, random_state=seed, stratify=strata
    )
    sample = sample.copy()

    rng = np.random.default_rng(seed)
    topped_up = [sample]
    for identity in IDENTITY_COLUMNS:
        current_count = int((sample[identity].fillna(0.0) >= DEFAULT_IDENTITY_THRESHOLD).sum())
        if current_count >= min_per_subgroup:
            continue
        pool = df[df[identity].fillna(0.0) >= DEFAULT_IDENTITY_THRESHOLD]
        pool = pool[~pool.index.isin(sample.index)]
        need = min_per_subgroup - current_count
        if len(pool) == 0:
            continue
        extra = pool.sample(n=min(need, len(pool)), random_state=int(rng.integers(0, 1_000_000)))
        topped_up.append(extra)

    result = pd.concat(topped_up).drop_duplicates(subset=[df.columns[0]] if len(df.columns) else None)
    # drop_duplicates on the id column if present, else on the whole row
    if "id" in df.columns:
        result = result.drop_duplicates(subset="id")
    return result.reset_index(drop=True)


def prepare_training_frame(
    df: pd.DataFrame,
    weighting: str = "metric_derived",
) -> pd.DataFrame:
    """Attach sample_weight column and binarized aux-subtype targets
    (filled 0 for rows missing annotations, matching how the competition's
    training kernels handle it -- the aux heads only get meaningful
    gradient on rows that have subtype annotations, which is fine since
    they're an auxiliary regularizer, not the scored output).

    `weighting`: one of {"none", "inverse_frequency", "metric_derived"}.
    """
    out = df.copy()
    if weighting == "none":
        out["sample_weight"] = 1.0
    elif weighting == "inverse_frequency":
        out["sample_weight"] = compute_inverse_frequency_weights(out)
    elif weighting == "metric_derived":
        out["sample_weight"] = compute_sample_weights(out)
    else:
        raise ValueError(f"unknown weighting scheme: {weighting}")

    for c in AUX_TOXICITY_COLUMNS:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out


if __name__ == "__main__":
    # Smoke test on synthetic data shaped like the real schema, so this
    # module is verifiable before the real 1.8M-row CSV is downloaded.
    rng = np.random.default_rng(0)
    n = 20_000
    toy = pd.DataFrame({"id": np.arange(n), "target": rng.random(n)})
    for c in IDENTITY_COLUMNS:
        # ~70% NaN (unannotated), else a fractional rater score
        vals = np.where(rng.random(n) < 0.7, np.nan, rng.random(n))
        toy[c] = vals
    for c in AUX_TOXICITY_COLUMNS:
        toy[c] = rng.random(n) * (toy["target"] >= 0.5)

    weighted = prepare_training_frame(toy, weighting="metric_derived")
    print(weighted["sample_weight"].describe())
    print("\nweight value counts (should be a handful of discrete levels):")
    print(weighted["sample_weight"].value_counts().sort_index())

    train_df, val_df = train_val_split(toy, val_size=0.1)
    print(f"\ntrain={len(train_df)} val={len(val_df)}")

    sub = stratified_subsample(toy, n=5000, min_per_subgroup=200)
    print(f"\nsubsample size={len(sub)}")
    for c in IDENTITY_COLUMNS:
        print(c, int((sub[c].fillna(0.0) >= 0.5).sum()))
