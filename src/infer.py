"""
Unified inference: turn raw text into scores, regardless of which model
produced them. Used by the bias probe, the error-analysis notebook, and
the FastAPI demo, so all three read a score the same way instead of each
reimplementing "load a checkpoint and run it."

Two backends:
- TfidfBaselineScorer: wraps the pickled Stage 0 model
  (models/baseline_tfidf.pkl written by src/baseline.py).
- TransformerScorer: wraps a MultiTaskToxicityModel checkpoint
  (models/*.pt written by src/train.py). Returns both the main toxicity
  score and, if the checkpoint has multi-task heads, the six auxiliary
  subtype scores.

Both expose the same `.score(texts: list[str]) -> np.ndarray` (and
TransformerScorer also `.score_with_aux(...)`), so callers don't need to
know which backend they're holding.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from scipy.special import expit as sigmoid

from metrics import AUX_TOXICITY_COLUMNS


class TfidfBaselineScorer:
    def __init__(self, pickle_path: str | Path):
        with open(pickle_path, "rb") as f:
            bundle = pickle.load(f)
        self.word_vec = bundle["word_vec"]
        self.char_vec = bundle["char_vec"]
        self.clf = bundle["clf"]

    def score(self, texts: list[str]) -> np.ndarray:
        Xw = self.word_vec.transform(texts)
        Xc = self.char_vec.transform(texts)
        X = hstack([Xw, Xc]).tocsr()
        return self.clf.predict_proba(X)[:, 1]


class TransformerScorer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "microsoft/deberta-v3-base",
        max_len: int = 220,
        use_multitask_heads: bool = True,
        device: str | None = None,
    ):
        import torch

        from model import MultiTaskToxicityModel
        from transformers import AutoTokenizer

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len
        self.model = MultiTaskToxicityModel(
            model_name=model_name,
            n_aux_labels=len(AUX_TOXICITY_COLUMNS),
            use_multitask_heads=use_multitask_heads,
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def score_with_aux(self, texts: list[str], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray | None]:
        torch = self._torch
        main_scores, aux_scores = [], []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=self.max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                main_logit, aux_logits = self.model(enc["input_ids"], enc["attention_mask"])
                main_scores.append(sigmoid(main_logit.cpu().numpy()))
                if aux_logits is not None:
                    aux_scores.append(sigmoid(aux_logits.cpu().numpy()))
        main = np.concatenate(main_scores)
        aux = np.concatenate(aux_scores) if aux_scores else None
        return main, aux

    def score(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        main, _ = self.score_with_aux(texts, batch_size)
        return main


def load_scorer(cfg: dict, prefer: str = "transformer"):
    """Convenience loader: try the transformer checkpoint first (if
    `prefer == 'transformer'`), fall back to the TF-IDF baseline pickle.
    Returns (scorer, backend_name) so callers (e.g. the demo) can report
    which model actually served the request.
    """
    root = Path(cfg["paths"]["model_dir"])
    transformer_ckpt = root / "deberta_multitask_final.pt"
    baseline_pkl = root / "baseline_tfidf.pkl"

    if prefer == "transformer" and transformer_ckpt.exists():
        tcfg = cfg["transformer"]
        return (
            TransformerScorer(
                transformer_ckpt,
                model_name=tcfg["model_name"],
                max_len=tcfg["max_len"],
                use_multitask_heads=tcfg["use_multitask_heads"],
            ),
            "transformer",
        )
    if baseline_pkl.exists():
        return TfidfBaselineScorer(baseline_pkl), "tfidf_baseline"
    raise FileNotFoundError(
        f"No model found in {root} (looked for deberta_multitask_final.pt "
        "and baseline_tfidf.pkl). Train one first: `make baseline` or `make train`."
    )


if __name__ == "__main__":
    # Smoke test: train a throwaway baseline on synthetic data, pickle it,
    # load it back through TfidfBaselineScorer, and confirm scores are
    # sane -- verifies the inference path end to end without needing the
    # real dataset or a GPU.
    import tempfile

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = np.random.default_rng(0)
    n = 500
    toxic_words = ["idiot", "stupid", "hate", "trash", "disgusting"]
    neutral_words = ["nice", "interesting", "today", "weather", "walk"]
    texts, labels = [], []
    for _ in range(n):
        is_toxic = rng.random() < 0.3
        words = rng.choice(toxic_words if is_toxic else neutral_words, size=6)
        texts.append(" ".join(words))
        labels.append(int(is_toxic))

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=500)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=500)
    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)
    X = hstack([Xw, Xc]).tocsr()
    clf = LogisticRegression(max_iter=1000).fit(X, labels)

    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "baseline_tfidf.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"word_vec": word_vec, "char_vec": char_vec, "clf": clf}, f)

        scorer = TfidfBaselineScorer(pkl_path)
        probe_texts = ["you are an idiot and I hate you", "what a nice walk today"]
        scores = scorer.score(probe_texts)
        print("scores:", dict(zip(probe_texts, scores)))
        assert scores[0] > scores[1], "toxic-worded text should score higher than neutral text"
        print("OK: TfidfBaselineScorer round-trips through pickle and ranks correctly.")
