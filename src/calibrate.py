"""
Probability calibration: temperature scaling, reliability diagrams, and
Expected Calibration Error (ECE).

Why this matters here: the threshold-selection section of the project
depends on "P(toxic | score=s)" meaning what it claims to mean. A model
can have excellent AUC (perfect ranking) while being badly miscalibrated
(e.g. everything above 0.3 is actually >90% likely toxic) -- AUC is
rank-invariant to any monotonic transform, so it cannot detect this at
all. If scores aren't calibrated, "sweep the threshold for 95% precision"
still works empirically (you're reading precision off real held-out data,
not off the score's face value), but the score can't be *interpreted* as
a probability for anything downstream (e.g. expected-cost calculations,
combining with other signals). We fit temperature scaling on logits and
report ECE/reliability before and after purely as a diagnostic.

Implemented with numpy + scipy only (no torch dependency) so this module
is runnable and testable without the heavy modeling stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _nll_at_temperature(temperature: float, logits: np.ndarray, labels: np.ndarray) -> float:
    scaled = logits / max(temperature, 1e-6)
    probs = sigmoid(scaled)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """`scores` are model probabilities in (0,1) (e.g. sigmoid outputs);
        internally converted to logits since temperature scaling is
        defined on logits: p' = sigmoid(logit(p) / T)."""
        logits = _logit(np.asarray(scores, dtype=np.float64))
        labels = np.asarray(labels, dtype=np.float64)
        result = minimize_scalar(
            _nll_at_temperature,
            bounds=(0.05, 10.0),
            args=(logits, labels),
            method="bounded",
        )
        self.temperature = float(result.x)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        logits = _logit(np.asarray(scores, dtype=np.float64))
        return sigmoid(logits / self.temperature)


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width-bin ECE: weighted average gap between
    predicted confidence and observed accuracy (here: observed positive
    rate, since this is a binary toxicity score) across bins.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        if not np.any(in_bin):
            continue
        bin_conf = scores[in_bin].mean()
        bin_acc = labels[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_diagram_data(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Returns per-bin (mean predicted score, observed positive rate,
    count) for plotting a reliability diagram. Plotting itself lives in
    the notebook / report generation code, not here, to keep this module
    dependency-light and unit-testable.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidence, bin_accuracy, bin_count = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        count = int(in_bin.sum())
        bin_count.append(count)
        if count == 0:
            bin_confidence.append(np.nan)
            bin_accuracy.append(np.nan)
        else:
            bin_confidence.append(float(scores[in_bin].mean()))
            bin_accuracy.append(float(labels[in_bin].mean()))
    return {
        "bin_edges": bin_edges,
        "bin_confidence": np.array(bin_confidence),
        "bin_accuracy": np.array(bin_accuracy),
        "bin_count": np.array(bin_count),
    }


if __name__ == "__main__":
    # Smoke test: synthesize an over-confident model (scores pushed toward
    # 0/1 relative to the true underlying probability) and confirm
    # temperature scaling brings ECE down.
    rng = np.random.default_rng(0)
    n = 20_000
    true_prob = rng.beta(2, 5, size=n)
    labels = (rng.random(n) < true_prob).astype(float)

    # Overconfident model: sharpen the true probability (push toward 0/1).
    sharpness = 3.0
    overconfident_logit = _logit(true_prob) * sharpness
    overconfident_scores = sigmoid(overconfident_logit + rng.normal(0, 0.1, n))

    ece_before = expected_calibration_error(overconfident_scores, labels, n_bins=10)
    scaler = TemperatureScaler().fit(overconfident_scores, labels)
    calibrated_scores = scaler.transform(overconfident_scores)
    ece_after = expected_calibration_error(calibrated_scores, labels, n_bins=10)

    print(f"fitted temperature: {scaler.temperature:.3f} (>1 means the model was overconfident, as constructed)")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")
    assert ece_after < ece_before, "temperature scaling should reduce ECE on this synthetic overconfident model"
    print("OK: calibration reduced ECE as expected.")
