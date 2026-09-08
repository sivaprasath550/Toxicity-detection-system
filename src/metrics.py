"""
Jigsaw "Unintended Bias in Toxicity Classification" evaluation metric.

Implemented from the competition's written definition (not copied from a
public kernel) so the reasoning is auditable end to end.

Background
----------
A single overall AUC hides subgroup-level bias: a model can separate toxic
from non-toxic comments very well in aggregate while systematically mis-
scoring comments that merely *mention* a protected identity. The competition
metric decomposes AUC into three views per identity subgroup, then combines
across subgroups with a generalized mean that punishes the worst subgroup
much harder than a plain average would.

Three subgroup-conditioned AUCs
--------------------------------
For a given identity subgroup S (e.g. "muslim"), let:
    - subgroup    = rows where the identity is mentioned (indicator == 1)
    - background  = rows where the identity is NOT mentioned

1. Subgroup AUC
   AUC computed using ONLY rows in `subgroup`. Answers: "within comments
   that mention this identity, can the model separate toxic from
   non-toxic?" A low value means the model is generally confused about
   this subgroup's comments (not specifically a false-positive/negative
   problem, just noisy).

2. BPSN AUC (Background Positive, Subgroup Negative)
   AUC computed on:
       - subgroup   rows that are NON-toxic  (labelled 0 / negative)
       - background rows that ARE   toxic    (labelled 1 / positive)
   This isolates the false-positive failure mode: benign subgroup mentions
   scoring as high as, or higher than, genuine toxicity elsewhere. This is
   the "I am a gay man" problem -- a low BPSN AUC means the identity term
   itself is acting as a toxicity signal.

3. BNSP AUC (Background Negative, Subgroup Positive)
   The mirror image:
       - subgroup   rows that ARE   toxic    (labelled 1 / positive)
       - background rows that are NON-toxic  (labelled 0 / negative)
   A low value means real attacks on the subgroup are being under-flagged
   relative to background toxicity.

Generalized power mean (p = -5)
--------------------------------
Given the per-subgroup scores for one of the three metrics above, combine
them with:

    M_p(x_1, ..., x_n) = ( mean(x_i ** p) ) ** (1 / p)

At p = -5 this behaves like a *soft minimum*: it is dominated by whichever
subgroup scores worst, so you cannot average your way to a good number by
doing well on eight subgroups and badly on one.

Final competition score
------------------------
    score = 0.25 * overall_AUC
           + 0.25 * power_mean(subgroup_AUC over all subgroups)
           + 0.25 * power_mean(BPSN_AUC over all subgroups)
           + 0.25 * power_mean(BNSP_AUC over all subgroups)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# The nine subgroups the competition actually scores on (large enough
# sample size in the identity-annotated subset to be statistically
# meaningful). Other identity columns exist in the raw data (hindu,
# buddhist, latino, ...) but are too sparse to score reliably -- keep them
# for error analysis, not for the headline metric.
IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]

# Six auxiliary toxicity subtypes used for the multi-task head, and useful
# on their own for error-mode analysis.
AUX_TOXICITY_COLUMNS = [
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
]

POWER_MEAN_P = -5


def power_mean(values, p: float = POWER_MEAN_P) -> float:
    """Generalized (Hölder) power mean, robust to zeros/negatives in `values`.

    For p < 0 this is undefined if any value is exactly 0 (division by
    zero inside the mean). AUCs are essentially never exactly 0 in
    practice; if it happens we clip to a tiny epsilon rather than raising,
    so one degenerate subgroup doesn't crash the whole evaluation.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    eps = 1e-12
    values = np.clip(values, eps, None)
    return float(np.mean(values**p) ** (1.0 / p))


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """roc_auc_score that returns NaN instead of raising when a split has
    only one class present (can happen on very small/sparse subgroups)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def subgroup_auc(df: pd.DataFrame, subgroup_col: str, label_col: str, pred_col: str) -> float:
    mask = df[subgroup_col] >= 0.5
    sub = df.loc[mask]
    return _safe_auc(sub[label_col].values, sub[pred_col].values)


def bpsn_auc(df: pd.DataFrame, subgroup_col: str, label_col: str, pred_col: str) -> float:
    subgroup_negative = df[(df[subgroup_col] >= 0.5) & (df[label_col] == 0)]
    background_positive = df[(df[subgroup_col] < 0.5) & (df[label_col] == 1)]
    combined = pd.concat([subgroup_negative, background_positive])
    return _safe_auc(combined[label_col].values, combined[pred_col].values)


def bnsp_auc(df: pd.DataFrame, subgroup_col: str, label_col: str, pred_col: str) -> float:
    subgroup_positive = df[(df[subgroup_col] >= 0.5) & (df[label_col] == 1)]
    background_negative = df[(df[subgroup_col] < 0.5) & (df[label_col] == 0)]
    combined = pd.concat([subgroup_positive, background_negative])
    return _safe_auc(combined[label_col].values, combined[pred_col].values)


@dataclass
class BiasMetricResult:
    per_subgroup: pd.DataFrame
    overall_auc: float
    subgroup_auc_power_mean: float
    bpsn_auc_power_mean: float
    bnsp_auc_power_mean: float
    final_score: float
    identity_columns: list = field(default_factory=list)

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "overall_auc": self.overall_auc,
                "subgroup_auc_power_mean": self.subgroup_auc_power_mean,
                "bpsn_auc_power_mean": self.bpsn_auc_power_mean,
                "bnsp_auc_power_mean": self.bnsp_auc_power_mean,
                "final_score": self.final_score,
            }
        )

    def worst_subgroups(self, metric: str = "bpsn_auc", n: int = 3) -> pd.DataFrame:
        return self.per_subgroup.sort_values(metric).head(n)


def compute_bias_metrics(
    df: pd.DataFrame,
    label_col: str = "target",
    pred_col: str = "prediction",
    identity_cols: list | None = None,
    label_threshold: float = 0.5,
    power_p: float = POWER_MEAN_P,
) -> BiasMetricResult:
    """Compute the full Jigsaw bias-metric report.

    Parameters
    ----------
    df : DataFrame containing, at minimum, `label_col`, `pred_col`, and one
        column per identity in `identity_cols`. `label_col` may be the raw
        soft fraction (e.g. 0.0-1.0 `target`) -- it is binarized here at
        `label_threshold`. Identity columns are expected in the same
        fractional form (proportion of raters who tagged that identity);
        rows with NaN identity values are treated as "identity not
        mentioned" (< 0.5) which matches how the competition scores the
        un-annotated 1.35M rows: they simply never enter any subgroup mask.
    label_col : soft or hard toxicity label column.
    pred_col : model score column (higher = more toxic).
    identity_cols : defaults to the 9 competition subgroups.
    label_threshold : binarization threshold for `label_col` (competition
        uses 0.5).
    power_p : exponent for the generalized mean (competition uses -5).

    Returns
    -------
    BiasMetricResult
    """
    if identity_cols is None:
        identity_cols = IDENTITY_COLUMNS

    missing = [c for c in identity_cols + [label_col, pred_col] if c not in df.columns]
    if missing:
        raise KeyError(f"compute_bias_metrics: missing columns {missing}")

    work = df.copy()
    # Binarize the label. NaN identity columns are filled with 0 so rows
    # without annotations simply never qualify for any subgroup mask
    # (they still count in "background" for BPSN/BNSP, which is correct:
    # the un-annotated 1.35M rows *are* the background population).
    work[label_col] = (work[label_col] >= label_threshold).astype(int)
    for c in identity_cols:
        work[c] = work[c].fillna(0.0)

    rows = []
    for identity in identity_cols:
        rows.append(
            {
                "subgroup": identity,
                "subgroup_size": int((work[identity] >= 0.5).sum()),
                "subgroup_auc": subgroup_auc(work, identity, label_col, pred_col),
                "bpsn_auc": bpsn_auc(work, identity, label_col, pred_col),
                "bnsp_auc": bnsp_auc(work, identity, label_col, pred_col),
            }
        )
    per_subgroup = pd.DataFrame(rows).set_index("subgroup")

    overall = _safe_auc(work[label_col].values, work[pred_col].values)

    sg_pm = power_mean(per_subgroup["subgroup_auc"].dropna().values, power_p)
    bpsn_pm = power_mean(per_subgroup["bpsn_auc"].dropna().values, power_p)
    bnsp_pm = power_mean(per_subgroup["bnsp_auc"].dropna().values, power_p)

    final_score = 0.25 * overall + 0.25 * sg_pm + 0.25 * bpsn_pm + 0.25 * bnsp_pm

    return BiasMetricResult(
        per_subgroup=per_subgroup,
        overall_auc=overall,
        subgroup_auc_power_mean=sg_pm,
        bpsn_auc_power_mean=bpsn_pm,
        bnsp_auc_power_mean=bnsp_pm,
        final_score=final_score,
        identity_columns=identity_cols,
    )


if __name__ == "__main__":
    # Self-test with synthetic data: no need for the real 1.8M-row dataset
    # to verify the metric implementation is wired correctly. We construct
    # a deliberately biased toy model (identity term -> inflated score) and
    # check that BPSN correctly tanks for the biased subgroup while a
    # matched "clean" subgroup scores well.
    rng = np.random.default_rng(0)
    n = 4000
    toxic = rng.integers(0, 2, size=n).astype(float)  # hard 0/1 "target"
    mentions_biased = rng.integers(0, 2, size=n).astype(float)  # e.g. "muslim"
    mentions_clean = rng.integers(0, 2, size=n).astype(float)  # e.g. "male"

    base_score = toxic * 0.35 + rng.normal(0, 0.15, n)
    # Biased model: mentioning the identity inflates the score regardless
    # of true toxicity -> should hurt BPSN for `mentions_biased`, since
    # non-toxic comments that merely mention it now compete on score with
    # genuinely toxic background comments.
    biased_pred = base_score + mentions_biased * 0.35
    clean_pred = base_score.copy()

    toy = pd.DataFrame(
        {
            "target": toxic,
            "muslim": mentions_biased,
            "male": mentions_clean,
            "prediction_biased": np.clip(biased_pred, 0, 1),
            "prediction_clean": np.clip(clean_pred, 0, 1),
        }
    )
    # fill the other 7 required identity columns with 0 (not mentioned)
    for c in IDENTITY_COLUMNS:
        if c not in toy.columns:
            toy[c] = 0.0

    for pred_col in ["prediction_biased", "prediction_clean"]:
        result = compute_bias_metrics(toy, pred_col=pred_col)
        print(f"\n=== {pred_col} ===")
        print(result.per_subgroup.loc[["muslim", "male"]])
        print(result.summary())

    print(
        "\nExpected: prediction_biased has a visibly lower bpsn_auc on "
        "'muslim' than on 'male' -- the identity term itself is inflating "
        "scores, so benign 'muslim' comments compete with genuinely toxic "
        "background comments and BPSN drops. prediction_clean carries the "
        "same real toxic-vs-not signal without the identity inflation, so "
        "it scores well and equally on both subgroups (no bias gap)."
    )
