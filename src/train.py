"""
Fine-tuning driver for the DeBERTa-v3 multi-task toxicity model.

Meant to run where a GPU is available (Kaggle/Colab) — see README for
timing (roughly 3.5-4.5h/epoch on a T4 at max_len=220, batch=32, over the
full 1.8M rows). Locally (CPU-only) use `--sample` for a structural
smoke test only; do not expect a real result from a CPU run.

Everything hyperparameter-shaped is read from config/config.yaml so the
weighting-scheme ablation (none / inverse_frequency / metric_derived) and
the multi-task-head ablation (on/off) are just config flips, not code
changes — that's what makes the ablation table in the README reproducible
from `make train CONFIG=...` rather than from hand-edited notebook cells.

Usage:
    python src/train.py --config config/config.yaml
    python src/train.py --config config/config.yaml --sample 2000  # smoke test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from data import load_raw, prepare_training_frame, train_val_split
from metrics import AUX_TOXICITY_COLUMNS, IDENTITY_COLUMNS, compute_bias_metrics
from model import MultiTaskToxicityModel, multitask_loss

ROOT = Path(__file__).resolve().parent.parent


class ToxicityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, aux_cols: list[str]):
        self.texts = df["comment_text"].fillna("").tolist()
        self.targets = df["target"].astype("float32").values
        self.weights = df["sample_weight"].astype("float32").values
        self.aux = df[aux_cols].astype("float32").values if all(c in df.columns for c in aux_cols) else None
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "sample_weight": torch.tensor(self.weights[idx], dtype=torch.float32),
        }
        if self.aux is not None:
            item["aux_target"] = torch.tensor(self.aux[idx], dtype=torch.float32)
        return item


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, device) -> np.ndarray:
    model.eval()
    all_logits = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        main_logit, _ = model(input_ids, attention_mask)
        all_logits.append(main_logit.detach().cpu().numpy())
    logits = np.concatenate(all_logits)
    return 1 / (1 + np.exp(-logits))  # sigmoid -> probability


def train(cfg: dict, sample: int | None = None, out_name: str = "model") -> None:
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    tcfg = cfg["transformer"]

    usecols = ["comment_text", "target"] + IDENTITY_COLUMNS + AUX_TOXICITY_COLUMNS
    raw_path = ROOT / cfg["paths"]["raw_train_csv"]
    df = load_raw(raw_path, usecols=usecols)
    df["comment_text"] = df["comment_text"].fillna("")

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=cfg["seed"]).reset_index(drop=True)
        print(f"[smoke test] subsampled to {len(df)} rows")

    train_df, val_df = train_val_split(df, val_size=cfg["data"]["val_size"], seed=cfg["seed"])
    train_df = prepare_training_frame(train_df, weighting=tcfg["weighting_scheme"])
    val_df = prepare_training_frame(val_df, weighting="none")  # weighting only applies to the training loss

    tokenizer = AutoTokenizer.from_pretrained(tcfg["model_name"])
    train_ds = ToxicityDataset(train_df, tokenizer, tcfg["max_len"], AUX_TOXICITY_COLUMNS)
    val_ds = ToxicityDataset(val_df, tokenizer, tcfg["max_len"], AUX_TOXICITY_COLUMNS)

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg["eval_batch_size"], shuffle=False, num_workers=2)

    model = MultiTaskToxicityModel(
        model_name=tcfg["model_name"],
        n_aux_labels=len(AUX_TOXICITY_COLUMNS),
        use_multitask_heads=tcfg["use_multitask_heads"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    total_steps = (len(train_loader) // tcfg["grad_accum_steps"]) * tcfg["epochs"]
    warmup_steps = int(total_steps * tcfg["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.amp.GradScaler(device.type, enabled=tcfg["fp16"] and device.type == "cuda")

    model_dir = ROOT / cfg["paths"]["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(tcfg["epochs"]):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            weight = batch["sample_weight"].to(device)
            aux_target = batch.get("aux_target")
            aux_target = aux_target.to(device) if aux_target is not None else None

            with torch.autocast(device_type=device.type, enabled=tcfg["fp16"] and device.type == "cuda"):
                main_logit, aux_logits = model(input_ids, attention_mask)
                loss = multitask_loss(
                    main_logit, target, weight, aux_logits, aux_target, tcfg["aux_loss_weight"]
                ) / tcfg["grad_accum_steps"]

            scaler.scale(loss).backward()
            running_loss += loss.item() * tcfg["grad_accum_steps"]

            if (step + 1) % tcfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % tcfg["log_every"] == 0:
                    elapsed = time.time() - t0
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss={running_loss / tcfg['log_every']:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} ({elapsed:.0f}s)")
                    running_loss = 0.0

                if global_step % tcfg["checkpoint_every_steps"] == 0:
                    # Overwrite a single "latest" file rather than one per
                    # step count -- a disconnect only ever needs the most
                    # recent checkpoint, and a multi-hour run checkpointing
                    # every few thousand steps would otherwise write dozens
                    # of full ~370MB state dicts (a real problem on Kaggle,
                    # where /kaggle/working has a bounded output size).
                    ckpt_path = model_dir / f"{out_name}_latest.pt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"checkpoint saved ({global_step} steps): {ckpt_path}")

        # End-of-epoch validation with the real bias metric, not just loss.
        val_scores = run_inference(model, val_loader, device)
        val_df_eval = val_df.copy()
        val_df_eval["prediction"] = val_scores
        result = compute_bias_metrics(val_df_eval, label_col="target", pred_col="prediction")
        print(f"\n=== epoch {epoch} validation ===")
        print(result.summary())
        print(result.per_subgroup)

    final_path = model_dir / f"{out_name}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nSaved final model to {final_path}")

    # Persist the LAST epoch's bias tables + full per-row val predictions --
    # consumed by notebooks/03_error_analysis.ipynb, src/thresholds.py, and
    # the README's result tables. Written under out_name so each ablation
    # run (e.g. deberta_multitask vs deberta_singlehead) gets its own files
    # instead of overwriting the previous run's numbers.
    out_dir = ROOT / cfg["paths"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    result.per_subgroup.to_csv(out_dir / f"{out_name}_per_subgroup.csv")
    result.summary().to_csv(out_dir / f"{out_name}_summary.csv")
    val_out_cols = ["comment_text", "target", "prediction"] + IDENTITY_COLUMNS
    val_df_eval[val_out_cols].to_csv(out_dir / f"{out_name}_val_predictions.csv", index=False)
    print(f"Saved bias tables + val predictions to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--sample", type=int, default=None, help="subsample rows for a smoke test")
    parser.add_argument("--out-name", type=str, default="deberta_multitask")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, sample=args.sample, out_name=args.out_name)
