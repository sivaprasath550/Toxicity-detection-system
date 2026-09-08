"""
DeBERTa-v3 encoder with a multi-task classification head.

Architecture rationale (see README "Stage 2" for the full argument): a
single toxicity head can satisfy its loss by learning "identity term
present -> toxic", because that shortcut correlates with the training
label often enough to lower the loss. Forcing the model to also predict
the 6 auxiliary subtypes (severe_toxicity, obscene, threat, insult,
identity_attack, sexual_explicit) makes that shortcut actively harmful:
a comment that just mentions an identity with no insult/threat/attack
content now gets a loss penalty on the aux heads if the model scores it
toxic-shaped, which pushes the encoder to separate "identity mentioned"
from "identity attacked". This is a regularizer that targets the exact
failure mode BPSN measures.

`use_multitask_heads=False` collapses this to a single-head baseline, so
the same class can produce both arms of the ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class MultiTaskToxicityModel(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        n_aux_labels: int = 6,
        use_multitask_heads: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_multitask_heads = use_multitask_heads
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.main_head = nn.Linear(hidden, 1)  # target (soft toxicity)
        if use_multitask_heads:
            self.aux_head = nn.Linear(hidden, n_aux_labels)
        else:
            self.aux_head = None

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Mean pooling over non-padding tokens. Simpler and generally more
        # robust than the raw [CLS]/first-token vector for this encoder,
        # and cheap.
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**encoder_kwargs)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        main_logit = self.main_head(pooled).squeeze(-1)  # (batch,)
        aux_logits = self.aux_head(pooled) if self.aux_head is not None else None  # (batch, n_aux)
        return main_logit, aux_logits


def multitask_loss(
    main_logit: torch.Tensor,
    main_target: torch.Tensor,
    sample_weight: torch.Tensor,
    aux_logits: torch.Tensor | None = None,
    aux_targets: torch.Tensor | None = None,
    aux_loss_weight: float = 0.25,
) -> torch.Tensor:
    """Weighted BCE on the main soft-label target, plus (optionally) an
    unweighted BCE on the 6 aux subtype heads. Main target and predictions
    are kept as soft probabilities (BCEWithLogits against the raw fraction
    in [0,1], not a binarized 0/1) -- this is the "free signal" the spec
    calls out: training against the rater-agreement fraction directly
    rather than throwing it away at the 0.5 threshold.
    """
    main_loss_per_example = nn.functional.binary_cross_entropy_with_logits(
        main_logit, main_target, reduction="none"
    )
    main_loss = (main_loss_per_example * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)

    if aux_logits is None or aux_targets is None:
        return main_loss

    aux_loss = nn.functional.binary_cross_entropy_with_logits(
        aux_logits, aux_targets, reduction="mean"
    )
    return main_loss + aux_loss_weight * aux_loss


if __name__ == "__main__":
    # Structural smoke test with a tiny model so this runs on CPU in
    # seconds without downloading deberta-v3-base. Swap model_name for the
    # real one when running with GPU access (Kaggle/Colab).
    tiny_model_name = "hf-internal-testing/tiny-random-DebertaV2Model"
    try:
        model = MultiTaskToxicityModel(model_name=tiny_model_name, use_multitask_heads=True)
    except Exception as e:
        print(f"Skipping model.py smoke test (no network access to fetch a tiny HF model): {e}")
    else:
        batch, seq_len = 4, 16
        input_ids = torch.randint(0, model.config.vocab_size, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        main_logit, aux_logits = model(input_ids, attention_mask)
        print("main_logit shape:", main_logit.shape)
        print("aux_logits shape:", None if aux_logits is None else aux_logits.shape)

        target = torch.rand(batch)
        weight = torch.ones(batch)
        aux_targets = torch.rand(batch, 6)
        loss = multitask_loss(main_logit, target, weight, aux_logits, aux_targets)
        print("loss:", loss.item())
