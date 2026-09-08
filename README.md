# Jigsaw Unintended Bias in Toxicity Classification

A toxicity classifier built to be evaluated on more than one number. Overall
AUC can look excellent while the model is unusable for specific identity
groups; this project measures that failure directly, tries several
mitigations, and reports what worked, what didn't, and what the honest
tradeoffs are.

> **Status: scaffolding complete, results pending a full training run.**
> The metric, weighting, calibration, threshold, and probe logic below are
> implemented and unit-verified against synthetic data (see `make test`).
> The tables in this README will be filled in with real numbers once the
> DeBERTa fine-tune has run on Kaggle/Colab against the actual dataset —
> see [Status / next steps](#status--next-steps) for exactly what's left.

## Results (fill in after training)

| Model | Overall AUC | Subgroup AUC (power mean) | BPSN AUC (power mean) | BNSP AUC (power mean) | Final score |
|---|---|---|---|---|---|
| TF-IDF + LogisticRegression (baseline) | TBD | TBD | TBD | TBD | TBD |
| DeBERTa-v3-base, single head, no weighting | TBD | TBD | TBD | TBD | TBD |
| DeBERTa-v3-base, multi-task heads, metric-derived weighting (final) | TBD | TBD | TBD | TBD | TBD |

**Counterfactual probe (before vs. after mitigation):** _bar chart goes here_
— mean toxicity score per identity term across 20 neutral templates
(`src/bias_probe.py`), e.g. "I am a **gay** man." vs. "I am a **tall** man."
A large, unexplained gap between an identity term and the control terms is
the clearest single piece of evidence of learned bias this project produces.

## Why not just report accuracy or F1@0.5

The base rate at the competition's toxicity threshold (target ≥ 0.5) is
about 8%. Accuracy on an 8%-positive problem is dominated by the majority
class and is close to meaningless here — a classifier that outputs "not
toxic" for everything scores ~92% accuracy. F1@0.5 is only marginally
better: it silently bakes in a threshold and an averaging choice nobody
asked for. See [Threshold selection](#threshold-selection-as-a-deployment-decision)
for what we report instead.

Overall AUC is a reasonable ranking metric but hides bias completely: a
model can separate toxic from non-toxic very well in aggregate while
scoring benign mentions of an identity ("I am a gay man") as toxic as real
attacks. The whole point of this project is to measure and address that,
which is why most of the effort below goes into the bias metric, not the
headline AUC.

## The dataset

[Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification)
(Kaggle, 2019), sourced from the Civil Comments platform. ~1.80M training
rows.

- **`target`** is a *fraction*, not a label — the proportion of human
  raters who marked the comment toxic. We train against this soft value
  directly (BCE against the raw fraction, not a binarized 0/1) and only
  binarize at ≥ 0.5 for evaluation, since that's free signal most public
  kernels throw away.
- **Identity annotations exist on ~450k of the 1.80M rows only** (the rest
  are NaN). All subgroup-level measurement in this project is necessarily
  restricted to that annotated subset; the other 1.35M rows still count as
  "background" (not-this-identity) in the BPSN/BNSP calculations, which is
  the correct interpretation — absence of annotation is not evidence of
  absence of identity, but it's all the evidence we have.
- **Nine subgroups are scored**: `male`, `female`,
  `homosexual_gay_or_lesbian`, `christian`, `jewish`, `muslim`, `black`,
  `white`, `psychiatric_or_mental_illness` — chosen (by the competition,
  not by us) because they have enough annotated examples to score
  reliably. Other identity columns in the raw data (`hindu`, `buddhist`,
  `latino`, ...) are too sparse to score but are kept for error analysis.
- **Six auxiliary toxicity subtypes**: `severe_toxicity`, `obscene`,
  `threat`, `insult`, `identity_attack`, `sexual_explicit`. Used as
  auxiliary training targets (see [Multi-task heads](#stage-2-multi-task-auxiliary-heads)).
- **Length distribution**: _histogram goes here_ (`notebooks/01_eda_spark.ipynb`) —
  `max_len=220` is chosen to cover ~99% of comments.

## The metric

Implemented from scratch in [`src/metrics.py`](src/metrics.py) (~250 lines
incl. docstrings) rather than copied from a public kernel, and verified
against a synthetic toy example in the same file's `__main__` block
(`python src/metrics.py`) before ever touching the real data.

For each of the 9 scored subgroups, three AUCs are computed:

1. **Subgroup AUC** — AUC restricted to rows that mention the identity.
   Can the model separate toxic from non-toxic *within* that population?
2. **BPSN AUC** (Background Positive, Subgroup Negative) — AUC over
   {non-toxic subgroup rows} ∪ {toxic background rows}. This is the
   **false-positive metric**: a low BPSN means benign mentions of the
   identity are scoring as high as, or higher than, genuine toxicity
   elsewhere. This is the number the whole project is about.
3. **BNSP AUC** (Background Negative, Subgroup Positive) — the mirror: a
   low value means real attacks on the subgroup are under-flagged.

These are combined with a **generalized power mean at p = −5** (a soft
minimum — dominated by the worst-scoring subgroup, so no amount of
across-the-board averaging can hide one badly-served group), and the final
competition score is:

```
score = 0.25 * overall_AUC
      + 0.25 * power_mean(subgroup_AUC, p=-5)
      + 0.25 * power_mean(BPSN_AUC, p=-5)
      + 0.25 * power_mean(BNSP_AUC, p=-5)
```

Verify the implementation: `python src/metrics.py`. It builds a toy
"biased" model where mentioning an identity inflates score regardless of
true toxicity, and confirms BPSN AUC for that identity collapses toward
0.5 (chance) while an identity-blind model shows no such gap.

## Class imbalance: what we do, and what we deliberately don't

We do **not** use SMOTE — it's not meaningful on transformer text
embeddings and doesn't address anything about *why* the model is biased.
We also do **not** naively oversample the toxic class: toxic comments in
this dataset disproportionately mention certain identities, so
oversampling toxic examples would amplify that correlation and make the
bias problem *worse*, not better.

Instead we use **metric-derived sample weighting** (`src/data.py:compute_sample_weights`),
built directly from the same four categories the bias metric cares about:

```
w  = 0.25                                          # base
w += 0.25   if comment mentions any of the 9 subgroups
w += 0.25   if toxic AND mentions no subgroup        # background positive
w += 0.25   if non-toxic AND mentions a subgroup      # subgroup negative
```

The last term is the important one: **benign comments that mention an
identity get upweighted**. Those are exactly the examples whose
misclassification tanks BPSN. This isn't a generic imbalance fix — it's
derived from the evaluation objective, which is the point.

| Weighting scheme | Overall AUC | BPSN AUC (power mean) | Notes |
|---|---|---|---|
| None | TBD | TBD | |
| Inverse-frequency class weighting | TBD | TBD | standard baseline, ignores identity structure |
| Metric-derived (ours) | TBD | TBD | |

## Model plan

### Stage 0 — TF-IDF + LogisticRegression baseline
`src/baseline.py`. Word (1-2 gram) + char (2-5 gram, `char_wb`) TF-IDF into
`LogisticRegression`. CPU, ~20 minutes on the full dataset. Not filler —
it's the number that makes "the transformer bought +X AUC but +Y BPSN" a
meaningful sentence instead of an unanchored one.

### Stage 1 — DeBERTa-v3-base
`microsoft/deberta-v3-base`, `max_len=220`, fp16, batch 32, lr 2e-5, linear
warmup over 5% of steps, AdamW, 1–2 epochs (this dataset overfits fast —
Kaggle winners mostly trained 1-2 epochs).

### Stage 2 — multi-task auxiliary heads
`src/model.py`. One head predicts `target` (soft label, BCE); six more
predict the auxiliary subtypes (BCE, weighted 0.25× the main loss). The
auxiliary task forces the encoder to distinguish "identity attack" from
"obscene" from "plain identity mention" — exactly the distinction that
stops it from collapsing "mentions an identity" into "toxic". It's a
regularizer that targets the bias metric directly, not a generic
multi-task convenience.

| | Overall AUC | BPSN AUC (power mean) | Notes |
|---|---|---|---|
| Single head | TBD | TBD | |
| Multi-task heads (ours) | TBD | TBD | |

## Threshold selection as a deployment decision

We never report a single F1@0.5. At an ~8% base rate a 0.5 threshold can
sit anywhere from mediocre to reasonable precision depending on the model,
and neither the threshold nor the resulting number means anything without
a stated policy. Instead (`src/thresholds.py`):

- **The target precision comes from the policy, not the data.** Auto-
  removal needs something like 0.95 precision (no human in the loop);
  routing to a human-review queue can tolerate ~0.60 because a person
  makes the final call.
- We **sweep the threshold on held-out validation** to find the threshold
  that *achieves* each target precision, and report the recall obtained
  there — reachability is checked explicitly (`threshold_for_precision`
  reports `reachable=False` if a target can't be hit on the data at hand).
- We repeat the sweep **per subgroup** (`per_subgroup_thresholds`), because
  a global threshold calibrated to 0.95 precision does not deliver 0.95
  precision on every subgroup — some identities are systematically
  over-scored, so the same cutoff is effectively stricter for them. We
  report that gap rather than average over it.
- This surfaces a genuine, unresolved tension: per-subgroup thresholds
  equalize error rates across groups, but mean identical text gets
  treated differently depending on which identity terms it contains —
  which is its own fairness problem. Naming the tension is the point;
  this project does not claim to resolve it.
- **Two thresholds, not one**: below `t_low` → allow, between `t_low` and
  `t_high` → route to human review, above `t_high` → auto-action
  (`two_threshold_routing`). The fraction of traffic landing in the review
  band is a headcount number, reported below.
- **Calibration** (`src/calibrate.py`): temperature scaling, reliability
  diagram, and Expected Calibration Error, before and after. A
  cost-based threshold isn't meaningful if the score it's applied to isn't
  calibrated as a probability.

| Target precision | Global threshold | Recall | Worst-subgroup precision at this threshold |
|---|---|---|---|
| 0.95 (auto-action) | TBD | TBD | TBD |
| 0.60 (human review) | TBD | TBD | TBD |

## Bias measurement and mitigation

**Measure.** Full 9-subgroup Subgroup/BPSN/BNSP AUC table for the
baseline model — _table goes here_. Expect the worst BPSN on `black`,
`white`, `homosexual_gay_or_lesbian`, and `muslim`: those terms appear
disproportionately in toxic training comments, so the baseline learns the
term itself as a toxicity signal.

**Demonstrate.** `src/bias_probe.py` builds 20 neutral templates × 9
identity terms + 4 control terms (180 + 80 sentences) — e.g. "I am a
{identity} man.", "My friend is {identity}." — and scores all of them.
None are toxic under any reasonable reading; a well-behaved model should
score them all low regardless of the identity term. The gap between an
identity term's mean score and the control-term mean is the headline
number — _bar chart goes here_.

**Mitigate**, more than one approach, compared honestly:
1. Metric-derived reweighting (above)
2. Multi-task auxiliary heads (above)
3. Counterfactual data augmentation — swap identity terms in non-toxic
   training comments to break the term↔toxicity correlation
4. *(stretch)* token-attribution check (integrated gradients / attention
   rollout) on a handful of false positives, before vs. after mitigation

**Re-measure.** Deltas reported honestly, including negative results —
e.g. if augmentation costs overall AUC for a small BPSN gain, that trade
is reported rather than dropped from the writeup.

| Mitigation | Δ Overall AUC | Δ BPSN AUC (power mean) | Kept? |
|---|---|---|---|
| + metric-derived weighting | TBD | TBD | TBD |
| + multi-task heads | TBD | TBD | TBD |
| + counterfactual augmentation | TBD | TBD | TBD |

## Error analysis

~200 errors sampled stratified across high-confidence false positives,
high-confidence false negatives, and near-threshold cases
(`notebooks/03_error_analysis.ipynb`), read and clustered into named
failure modes:

| Failure mode | Count | Example | What we'd do about it |
|---|---|---|---|
| Reclaimed slurs / in-group speech | TBD | TBD | TBD |
| Quoted toxicity (reporting abuse gets flagged) | TBD | TBD | TBD |
| Sarcasm / irony | TBD | TBD | TBD |
| Counter-speech (arguing against racism flagged as racism) | TBD | TBD | TBD |
| Label noise / annotator disagreement | TBD | TBD | TBD |
| Politics-as-toxicity | TBD | TBD | TBD |

If a meaningful fraction of "errors" turn out to be annotation
disagreements rather than model mistakes, that's reported explicitly — it
lowers the model's real ceiling and is worth saying out loud.

## Where Spark fits (and where it doesn't)

PySpark is used, in **local mode**, for what it's actually good for at
this scale (`notebooks/01_eda_spark.ipynb`, `make spark-eda`):
- loading/profiling the 1.8M rows (schema, length distribution, subgroup
  counts, per-subgroup base rates)
- a text-normalization/tokenization UDF pass, written to Parquet
- (batch-inference simulation over a partitioned set is the natural next
  extension once a trained model exists — not yet wired up)

**Local mode at this scale is a demonstration of the API, not a
distributed workload** — the same job would run unchanged on a cluster,
but no claim of distributed computing is being made here.

*Windows note (this dev machine): PySpark needs a JVM and Hadoop's
`winutils.exe`/`hadoop.dll` shims to do any local filesystem I/O
(read CSV, write Parquet) — neither ships with `pip install pyspark`.
This repo's dev environment has `JAVA_HOME` (Temurin 17), `HADOOP_HOME`
(winutils 3.3.6, from `cdarlint/winutils`), and `PYSPARK_PYTHON` set as
user-level environment variables so `make spark-eda` works out of the
box here; a fresh machine (or Linux/Mac) needs the equivalent JVM +
(on Windows only) winutils setup once.*

## Repo structure

```
jigsaw-toxicity-bias/
├── README.md
├── app/
│   └── main.py            # FastAPI demo: score, subtype breakdown, routing decision
├── notebooks/
│   ├── 01_eda_spark.ipynb
│   ├── 02_baseline_tfidf.ipynb
│   ├── 03_error_analysis.ipynb
│   └── kaggle/train_deberta_kaggle.ipynb   # self-contained Kaggle-GPU notebook
├── scripts/
│   └── gen_kaggle_notebook.py   # regenerates the Kaggle notebook from src/
├── src/
│   ├── data.py            # loading, weighting scheme, splits/subsampling
│   ├── metrics.py         # subgroup/BPSN/BNSP + power mean
│   ├── model.py           # DeBERTa + multi-task heads
│   ├── train.py           # training loop (local CPU smoke test or Kaggle/Colab GPU)
│   ├── baseline.py        # Stage 0 TF-IDF + LogisticRegression
│   ├── infer.py           # unified scoring: TF-IDF pickle or transformer checkpoint
│   ├── calibrate.py       # temperature scaling, ECE, reliability diagram
│   ├── thresholds.py      # threshold sweep, per-subgroup, two-threshold routing
│   ├── bias_probe.py      # counterfactual identity-swap templates
│   └── error_analysis.py  # stratified error sampling, label-noise heuristic
├── reports/
│   ├── figures/
│   └── results.md
├── config/config.yaml
├── requirements.txt
└── Makefile
```

## Reproducing this

```bash
pip install -r requirements.txt

# 1. Get the data (needs a Kaggle account + API token at ~/.kaggle/kaggle.json)
make data

# 2. Sanity-check every module against synthetic data (no download needed)
make test

# 3. Real EDA (length distribution, subgroup base rates) -- needs a JVM, see above
make spark-eda

# 4. Stage 0 baseline (CPU, ~20 min on the full data; use SAMPLE=50000 to iterate faster)
make baseline

# 5. Stage 1/2 fine-tune -- generate + run the Kaggle notebook (GPU; see Compute budget)
make kaggle-notebook   # regenerate notebooks/kaggle/train_deberta_kaggle.ipynb from src/
# upload that notebook to a Kaggle Notebook, attach the competition dataset, run it

# 6. Error analysis once a run's val predictions exist under reports/
make error-analysis

# 7. Try the demo against whatever model is trained so far
make demo   # -> http://localhost:8000
```

## Compute budget

Rough numbers on a T4, fp16, `max_len=220`, batch 32: **~3.5-4.5h/epoch**
on the full 1.8M rows. Kaggle sessions run 9h (12h on TPU), so this fits
but leaves little margin — develop and iterate on a 200k stratified
dev subsample (`src/data.py:stratified_subsample`, ~25 min/loop), then run
2-3 full-data runs at the end for the numbers reported here. Checkpoints
are written every `checkpoint_every_steps` steps
(`config/config.yaml:transformer.checkpoint_every_steps`) so a disconnect
doesn't cost a full run.

## Limitations

- **Civil Comments is news-comment English** — not multilingual, not
  representative of e.g. professional/workplace text. Nothing here
  generalizes to other domains or languages without re-validation.
- **Annotator labels encode annotator bias.** The `target` fraction is
  human-rater agreement, not ground truth; the error-analysis section
  quantifies how much of the model's apparent error is actually
  disagreement with annotators, which caps how good "the real number" can
  ever look.
- **The 9 scored subgroups are a coarse proxy for identity** — chosen by
  sample size, not by who is most affected by moderation failures in
  practice.
- **Fairness metrics conflict with each other.** This project optimizes
  one family (subgroup/BPSN/BNSP AUC via the power mean); per-subgroup
  threshold equalization is a different, and partly incompatible,
  fairness criterion. Both are discussed above; neither is "solved."
- **This is a portfolio project, not a production system.** No claim of
  distributed training/serving, no live traffic, no online monitoring.

## Status / next steps

Every piece of code in this repo is implemented and **verified against
synthetic data shaped like the real schema** — not just written, actually
run: the bias metric, the weighting derivation, calibration, the threshold
sweep, the counterfactual probe generator, the TF-IDF baseline, the full
DeBERTa multi-task training loop (tokenizer → training steps → scheduler →
checkpointing → epoch-end bias-metric validation), the inference layer,
the FastAPI demo, the error-analysis sampler, and the real PySpark EDA
pipeline (CSV → length histograms with the actual DeBERTa tokenizer →
subgroup base rates → tokenize-UDF → Parquet, including getting a JVM +
Hadoop's Windows shims working on this machine). `make test` runs the
seven module-level checks; nothing else needs the real dataset to verify.

**The one remaining blocker is the Kaggle dataset itself**, and it's a
genuine one, not a to-do: downloading it requires *this user's* Kaggle
account (competition rules acceptance + an API token), which nothing in
this repo can substitute for. Once `~/.kaggle/kaggle.json` exists:

1. `make data` → the real `train.csv`.
2. `make spark-eda` → real length distribution, subgroup counts, base rates.
3. `make baseline` on the real data → fill in the Stage 0 row above.
4. Upload `notebooks/kaggle/train_deberta_kaggle.ipynb` to a Kaggle
   Notebook (GPU on, competition dataset attached, internet on) → run the
   single-head/no-weighting arm first (Stage 1 row), then the multi-task +
   metric-derived-weighting arm (final row); flip the two ablation knobs
   in the notebook's config cell and re-run for the ablation tables.
5. Score the counterfactual probe set with the trained model → bar chart.
6. `make error-analysis` (or open the notebook) once a run's
   `*_val_predictions.csv` exists under `reports/` → read the sample,
   fill in the failure-mode table.
7. `make demo` already works today against the TF-IDF baseline and
   upgrades itself automatically once a transformer checkpoint lands at
   `models/deberta_multitask_final.pt`.

## Resume line (fill in once trained)

> Fine-tuned DeBERTa-v3 with auxiliary-subtype multi-task heads on 1.8M
> Civil Comments; raised worst-subgroup BPSN AUC from **X** to **Y** at
> ≤**Z** overall AUC cost via metric-derived loss reweighting; per-subgroup
> threshold calibration at fixed 95% precision.
