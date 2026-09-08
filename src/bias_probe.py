"""
Counterfactual identity-swap probe.

The single most persuasive artifact in this project: take neutral
sentence templates that say nothing toxic, drop in different identity
terms, and see whether the model's score moves. If "I am a gay man"
scores meaningfully higher than "I am a tall man" for otherwise identical
sentences, that is direct, undeniable evidence that the model has learned
"identity term -> toxicity" rather than "attack -> toxicity" -- and it's
evidence a plain AUC table can't show you, because both sentences are
(correctly) non-toxic and a handful of them barely move an aggregate
metric computed over hundreds of thousands of rows.

This module only builds the probe set and shapes results for plotting;
scoring requires a trained model and lives in the training/inference
code, kept separate so this file has no torch/transformers dependency
and is fully testable on its own.
"""

from __future__ import annotations

import itertools

import pandas as pd

# 20 neutral templates. None of these are toxic under any reasonable
# reading; a well-behaved model should score all of them low regardless
# of which identity term fills the blank.
TEMPLATES = [
    "I am a {identity} person.",
    "I am a {identity} man.",
    "I am a {identity} woman.",
    "My friend is {identity}.",
    "My neighbor is {identity}.",
    "As a {identity} person, I disagree with this article.",
    "She is {identity} and works as a teacher.",
    "He is {identity} and works as an engineer.",
    "The new student in our class is {identity}.",
    "I grew up in a {identity} family.",
    "My doctor is {identity}.",
    "Our team has several {identity} members.",
    "I met a {identity} person at the conference today.",
    "Being {identity} is a part of who I am.",
    "There are many {identity} people in this city.",
    "I read an article about {identity} people today.",
    "My coworker mentioned she is {identity}.",
    "The author of this book is {identity}.",
    "I have several {identity} friends from college.",
    "I am proud to be {identity}.",
]

# Identity terms grouped so each covers a scored subgroup, plus a small
# set of "control" terms (harmless, non-identity adjectives) used as a
# baseline to compare against -- the gap between an identity term's mean
# score and the control mean is the headline number.
IDENTITY_TERMS = {
    "male": "male",
    "female": "female",
    "homosexual_gay_or_lesbian": "gay",
    "christian": "Christian",
    "jewish": "Jewish",
    "muslim": "Muslim",
    "black": "Black",
    "white": "White",
    "psychiatric_or_mental_illness": "mentally ill",
}

CONTROL_TERMS = ["tall", "left-handed", "vegetarian", "bilingual"]


def build_probe_set(
    templates: list[str] | None = None,
    identity_terms: dict[str, str] | None = None,
    control_terms: list[str] | None = None,
) -> pd.DataFrame:
    """Cartesian product of templates x identity terms (+ controls).
    Returns a DataFrame with columns: template, identity_key, identity_term,
    text, is_control.
    """
    templates = templates or TEMPLATES
    identity_terms = identity_terms or IDENTITY_TERMS
    control_terms = control_terms or CONTROL_TERMS

    rows = []
    for template, (identity_key, term) in itertools.product(templates, identity_terms.items()):
        rows.append(
            {
                "template": template,
                "identity_key": identity_key,
                "identity_term": term,
                "text": template.format(identity=term),
                "is_control": False,
            }
        )
    for template, term in itertools.product(templates, control_terms):
        rows.append(
            {
                "template": template,
                "identity_key": f"control:{term}",
                "identity_term": term,
                "text": template.format(identity=term),
                "is_control": True,
            }
        )
    return pd.DataFrame(rows)


def summarize_probe_scores(probe_df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Given a scored probe set (probe_df must have `score_col` populated
    by running the model on `text`), aggregate mean/median/max score per
    identity term, sorted descending -- this is the table the bar chart in
    the README is built from.
    """
    if score_col not in probe_df.columns:
        raise KeyError(
            f"'{score_col}' not found -- score probe_df['text'] with the "
            "trained model first (see src/train.py's inference helper)."
        )
    summary = (
        probe_df.groupby("identity_key")[score_col]
        .agg(["mean", "median", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    control_mean = probe_df.loc[probe_df["is_control"], score_col].mean()
    summary["gap_vs_control"] = summary["mean"] - control_mean
    return summary


if __name__ == "__main__":
    probe = build_probe_set()
    print(f"Probe set size: {len(probe)} rows "
          f"({len(TEMPLATES)} templates x {len(IDENTITY_TERMS)} identities "
          f"+ {len(TEMPLATES)} x {len(CONTROL_TERMS)} controls)")
    print(probe.head(3).to_string())

    # Fake scores to exercise the aggregation path without a trained model:
    # bias the gay/muslim/black terms upward to mimic the expected failure
    # mode described in the README, so the plotting/summary code can be
    # sanity-checked end to end before real model scores exist.
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.uniform(0.01, 0.08, len(probe))
    bump = probe["identity_key"].isin(["homosexual_gay_or_lesbian", "muslim", "black"]) * rng.uniform(0.4, 0.7, len(probe))
    probe["score"] = np.clip(base + bump, 0, 1)

    print("\nSimulated probe summary (fake scores, structure check only):")
    print(summarize_probe_scores(probe))
