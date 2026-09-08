"""
Minimal moderation demo service.

Text box -> toxicity score -> per-subtype breakdown (if the loaded model
has multi-task heads) -> routing decision (allow / human_review /
auto_remove) using the two-threshold policy from config.yaml.

This exists to make the project tangible, not to replace the analysis --
see README "Optional: the demo". It loads whatever model is available
(transformer checkpoint if trained, else the TF-IDF baseline) via
src/infer.py, so it works today against the baseline and upgrades itself
automatically once a transformer checkpoint exists at
models/deberta_multitask_final.pt.

Run: uvicorn app.main:app --reload --port 8000   (or `make demo`)
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from infer import load_scorer, TransformerScorer  # noqa: E402
from metrics import AUX_TOXICITY_COLUMNS  # noqa: E402
from thresholds import two_threshold_routing  # noqa: E402

with open(ROOT / "config" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

app = FastAPI(title="Toxicity moderation demo")

_state: dict = {"scorer": None, "backend": None}


def get_scorer():
    if _state["scorer"] is None:
        try:
            scorer, backend = load_scorer(CONFIG, prefer="transformer")
        except FileNotFoundError as e:
            raise RuntimeError(
                "No trained model found. Run `make baseline` (fastest) or "
                "`make train` first."
            ) from e
        _state["scorer"], _state["backend"] = scorer, backend
    return _state["scorer"], _state["backend"]


class ScoreRequest(BaseModel):
    text: str
    # Operating thresholds are configurable per-request so the demo can
    # show how the routing decision shifts with policy, not just with text.
    t_low: float = 0.3
    t_high: float = 0.8


class ScoreResponse(BaseModel):
    text: str
    backend: str
    toxicity_score: float
    subtype_scores: dict[str, float] | None
    routing_decision: str
    thresholds_used: dict[str, float]


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    scorer, backend = get_scorer()

    subtype_scores = None
    if isinstance(scorer, TransformerScorer):
        main, aux = scorer.score_with_aux([req.text])
        main_score = float(main[0])
        if aux is not None:
            subtype_scores = {name: float(v) for name, v in zip(AUX_TOXICITY_COLUMNS, aux[0])}
    else:
        main_score = float(scorer.score([req.text])[0])

    decision = two_threshold_routing([main_score], t_low=req.t_low, t_high=req.t_high)[0]

    return ScoreResponse(
        text=req.text,
        backend=backend,
        toxicity_score=main_score,
        subtype_scores=subtype_scores,
        routing_decision=decision,
        thresholds_used={"t_low": req.t_low, "t_high": req.t_high},
    )


@app.get("/health")
def health():
    try:
        _, backend = get_scorer()
        return {"status": "ok", "backend": backend}
    except RuntimeError as e:
        return {"status": "no_model", "detail": str(e)}


_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Toxicity moderation demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  textarea { width: 100%; height: 100px; font-size: 14px; padding: 8px; box-sizing: border-box; }
  button { padding: 8px 16px; font-size: 14px; margin-top: 8px; cursor: pointer; }
  #result { margin-top: 20px; padding: 16px; border-radius: 8px; border: 1px solid #ddd; display: none; }
  .allow { background: #eafaf0; border-color: #8fd9a8; }
  .human_review { background: #fff8e1; border-color: #ffd54f; }
  .auto_remove { background: #fdecea; border-color: #f5a3a3; }
  table { width: 100%; margin-top: 8px; font-size: 13px; border-collapse: collapse; }
  td, th { text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; }
  .bar { background: #ccc; height: 8px; border-radius: 4px; overflow: hidden; }
  .bar-fill { background: #555; height: 100%; }
</style>
</head>
<body>
  <h2>Toxicity moderation demo</h2>
  <p style="color:#666; font-size: 13px;">
    Backend model, per-subtype breakdown, and routing decision (allow /
    human_review / auto_remove) computed from the two-threshold policy.
    See the project README for what these numbers mean and don't mean.
  </p>
  <textarea id="text" placeholder="Type a comment to score..."></textarea>
  <br>
  <button onclick="scoreText()">Score</button>
  <div id="result"></div>

<script>
async function scoreText() {
  const text = document.getElementById('text').value;
  const resultDiv = document.getElementById('result');
  resultDiv.style.display = 'block';
  resultDiv.className = '';
  resultDiv.innerHTML = 'Scoring...';
  try {
    const res = await fetch('/score', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    if (!res.ok) { resultDiv.innerHTML = 'Error: ' + JSON.stringify(data); return; }
    resultDiv.className = data.routing_decision;
    let html = `<b>Toxicity score:</b> ${data.toxicity_score.toFixed(3)}<br>`;
    html += `<b>Routing decision:</b> ${data.routing_decision} `;
    html += `(t_low=${data.thresholds_used.t_low}, t_high=${data.thresholds_used.t_high})<br>`;
    html += `<b>Model:</b> ${data.backend}<br>`;
    if (data.subtype_scores) {
      html += '<table><tr><th>Subtype</th><th>Score</th></tr>';
      for (const [k, v] of Object.entries(data.subtype_scores)) {
        const pct = Math.round(v * 100);
        html += `<tr><td>${k}</td><td>${v.toFixed(3)}<div class="bar"><div class="bar-fill" style="width:${pct}%"></div></div></td></tr>`;
      }
      html += '</table>';
    }
    resultDiv.innerHTML = html;
  } catch (e) {
    resultDiv.innerHTML = 'Request failed: ' + e;
  }
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return _PAGE
