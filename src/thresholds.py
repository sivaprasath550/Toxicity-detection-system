"""
Threshold selection as a deployment decision, not a leaderboard metric.

We deliberately never report a single F1@0.5. At an ~8% base rate, a
threshold picked by "looks reasonable" can sit anywhere from 40% to 90%
precision depending on the model, and neither the threshold nor the
number means anything without a stated policy. Instead:

- The operating precision comes from the policy (auto-remove needs very
  high precision because there's no human in the loop; a review queue can
  tolerate much lower precision because a person makes the final call).
- We sweep the threshold on a held-out split to find the threshold that
  *achieves* each target precision, and report the recall we get there.
- We repeat the sweep per subgroup, because a global threshold at 0.95
  precision will not deliver 0.95 precision on every subgroup -- some
  identities are systematically over-scored (see bias_probe.py), so the
  same cutoff is stricter for them in effect. We report that gap rather
  than paper over it; per-subgroup thresholds are one way to close it, at
  the cost of treating identical text differently depending on the
  identity terms it contains -- a genuine, unresolved tension, not a bug
  to be fixed away.
- We define two thresholds, not one: below t_low, auto-allow; between
  t_low and t_high, route to human review; above t_high, auto-action.
  The fraction of traffic landing in the review band is itself a
  reportable number (it's a headcount requirement).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def threshold_for_precision(
    labels: np.ndarray, scores: np.ndarray, target_precision: float
) -> dict:
    """Smallest threshold whose precision-recall-curve precision is >=
    target_precision, i.e. the most permissive (highest-recall) threshold
    that still satisfies the precision requirement.

    Returns dict(threshold, precision, recall) or, if the target
    precision is unreachable on this data (small/degenerate splits), the
    highest-precision point available with a `reachable=False` flag.
    """
    precision, recall, thresh = precision_recall_curve(labels, scores)
    # precision_recall_curve returns precision/recall with one more entry
    # than thresh (the last point is precision=1, recall=0 with no
    # threshold); align by dropping that last point.
    precision, recall = precision[:-1], recall[:-1]

    ok = precision >= target_precision
    if not np.any(ok):
        best_idx = int(np.argmax(precision))
        return {
            "threshold": float(thresh[best_idx]),
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "reachable": False,
        }
    # Among thresholds meeting the target, take the one with highest
    # recall (precision_recall_curve is ordered by increasing threshold,
    # i.e. decreasing recall, so among `ok` indices the first — lowest
    # threshold — gives the highest recall).
    candidate_idxs = np.where(ok)[0]
    best_idx = candidate_idxs[np.argmax(recall[candidate_idxs])]
    return {
        "threshold": float(thresh[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
        "reachable": True,
    }


def per_subgroup_thresholds(
    df: pd.DataFrame,
    identity_cols: list[str],
    label_col: str,
    score_col: str,
    target_precision: float,
    label_threshold: float = 0.5,
    identity_threshold: float = 0.5,
) -> pd.DataFrame:
    """Threshold-for-precision computed on the global population and on
    each identity subgroup separately, so the gap between "global
    threshold's precision on this subgroup" and "this subgroup's own
    threshold" is visible in one table.
    """
    labels_bin = (df[label_col] >= label_threshold).astype(int).values
    global_result = threshold_for_precision(labels_bin, df[score_col].values, target_precision)
    global_threshold = global_result["threshold"]

    rows = [{"subgroup": "GLOBAL", **global_result, "precision_at_global_threshold": global_result["precision"]}]
    for identity in identity_cols:
        mask = df[identity].fillna(0.0) >= identity_threshold
        sub_labels = labels_bin[mask.values]
        sub_scores = df.loc[mask, score_col].values
        if len(np.unique(sub_labels)) < 2 or mask.sum() < 20:
            rows.append(
                {
                    "subgroup": identity,
                    "threshold": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "reachable": False,
                    "precision_at_global_threshold": np.nan,
                }
            )
            continue
        own = threshold_for_precision(sub_labels, sub_scores, target_precision)
        # precision the GLOBAL threshold actually achieves on this subgroup
        predicted_positive = sub_scores >= global_threshold
        if predicted_positive.sum() > 0:
            precision_at_global = float((sub_labels[predicted_positive] == 1).mean())
        else:
            precision_at_global = float("nan")
        rows.append({"subgroup": identity, **own, "precision_at_global_threshold": precision_at_global})

    return pd.DataFrame(rows).set_index("subgroup")


def two_threshold_routing(
    scores: np.ndarray, t_low: float, t_high: float
) -> pd.Series:
    """Classify each score into allow / review / remove given the two
    operating thresholds, and return the resulting routing decision per
    row plus (via .value_counts() on the caller's side) the queue size.
    """
    scores = np.asarray(scores)
    decisions = np.where(scores >= t_high, "auto_remove", np.where(scores >= t_low, "human_review", "allow"))
    return pd.Series(decisions)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20_000
    labels = (rng.random(n) < 0.08).astype(int)  # ~8% base rate, matches the real dataset
    scores = np.clip(labels * rng.beta(5, 2, n) + (1 - labels) * rng.beta(2, 8, n), 0, 1)

    for target in [0.95, 0.60]:
        result = threshold_for_precision(labels, scores, target)
        print(f"target_precision={target}: {result}")

    df = pd.DataFrame({"target": labels.astype(float), "score": scores})
    df["muslim"] = np.where(rng.random(n) < 0.05, 1.0, 0.0)
    # simulate the biased-scoring effect: muslim-tagged rows get inflated scores
    df.loc[df["muslim"] == 1, "score"] = np.clip(df.loc[df["muslim"] == 1, "score"] + 0.25, 0, 1)

    table = per_subgroup_thresholds(df, ["muslim"], "target", "score", target_precision=0.95)
    print("\nPer-subgroup threshold table (0.95 target):")
    print(table)

    routing = two_threshold_routing(scores, t_low=0.3, t_high=0.8)
    print("\nRouting distribution:")
    print(routing.value_counts(normalize=True))
