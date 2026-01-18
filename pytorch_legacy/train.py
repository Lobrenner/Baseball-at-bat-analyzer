# src/train.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class LabelEncoder:
    classes_: List[str]
    to_idx: Dict[str, int]

    @classmethod
    def fit(cls, values: pd.Series, unk_token: str = "__UNK__") -> "LabelEncoder":
        uniq = [unk_token] + sorted({str(v) for v in values.dropna().astype(str).unique()})
        return cls(classes_=uniq, to_idx={c: i for i, c in enumerate(uniq)})

    def transform(self, values: pd.Series) -> np.ndarray:
        unk = self.to_idx["__UNK__"]
        return np.array([self.to_idx.get(str(v), unk) for v in values.astype(str)], dtype=np.int64)

    def to_jsonable(self) -> dict:
        return {"classes": self.classes_}

    @classmethod
    def from_jsonable(cls, obj: dict) -> "LabelEncoder":
        classes = obj["classes"]
        return cls(classes_=classes, to_idx={c: i for i, c in enumerate(classes)})


class PitchSenseDataset(Dataset):
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray):
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class PitchSenseMLP(nn.Module):
    def __init__(
        self,
        cat_sizes: List[int],
        num_numeric: int,
        num_classes: int,
        emb_dim: int = 16,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in cat_sizes])

        in_dim = emb_dim * len(cat_sizes) + num_numeric
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.mlp(x)


def split_train_val(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - val_frac))
    train_idx, val_idx = idx[:cut], idx[cut:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Train PitchSense v0 next-pitch model (PyTorch).")
    parser.add_argument("--data", default="data/processed/pitchsense_v0.parquet")
    parser.add_argument("--outdir", default="models/pitchsense_v0")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.data)
    print("[LOAD]", df.shape)

    cat_cols = ["stand", "p_throws", "prev_pitch_type_1", "prev_pitch_type_2"]
    num_cols = ["balls", "strikes"]
    target_col = "next_pitch_type"

    train_df, val_df = split_train_val(df, val_frac=args.val_frac, seed=args.seed)
    print("[SPLIT] train=", train_df.shape, "val=", val_df.shape)

    # Encoders fit on TRAIN only
    encoders: Dict[str, LabelEncoder] = {c: LabelEncoder.fit(train_df[c]) for c in cat_cols}
    y_enc = LabelEncoder.fit(train_df[target_col])

    # Transform categoricals
    X_train_cat = np.column_stack([encoders[c].transform(train_df[c]) for c in cat_cols])
    X_val_cat = np.column_stack([encoders[c].transform(val_df[c]) for c in cat_cols])

    # Numeric features (simple scaling)
    X_train_num = train_df[num_cols].to_numpy(dtype=np.float32)
    X_val_num = val_df[num_cols].to_numpy(dtype=np.float32)

    # Scale to ~[0,1]
    X_train_num[:, 0] /= 3.0  # balls
    X_train_num[:, 1] /= 2.0  # strikes
    X_val_num[:, 0] /= 3.0
    X_val_num[:, 1] /= 2.0

    y_train = y_enc.transform(train_df[target_col])
    y_val = y_enc.transform(val_df[target_col])

    train_ds = PitchSenseDataset(X_train_cat, X_train_num, y_train)
    val_ds = PitchSenseDataset(X_val_cat, X_val_num, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    cat_sizes = [len(encoders[c].classes_) for c in cat_cols]
    num_classes = len(y_enc.classes_)

    model = PitchSenseMLP(
        cat_sizes=cat_sizes,
        num_numeric=len(num_cols),
        num_classes=num_classes,
        emb_dim=16,
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        tr_batches = 0

        for Xc, Xn, yb in train_loader:
            Xc, Xn, yb = Xc.to(device), Xn.to(device), yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(Xc, Xn)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_acc += batch_accuracy(logits, yb)
            tr_batches += 1

        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        va_batches = 0

        with torch.no_grad():
            for Xc, Xn, yb in val_loader:
                Xc, Xn, yb = Xc.to(device), Xn.to(device), yb.to(device)
                logits = model(Xc, Xn)
                loss = loss_fn(logits, yb)

                va_loss += loss.item()
                va_acc += batch_accuracy(logits, yb)
                va_batches += 1

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={tr_loss/max(tr_batches,1):.4f} train_acc={tr_acc/max(tr_batches,1):.4f} | "
            f"val_loss={va_loss/max(va_batches,1):.4f} val_acc={va_acc/max(va_batches,1):.4f}"
        )

    # Save model weights
    model_path = os.path.join(args.outdir, "model.pt")
    torch.save({"state_dict": model.state_dict(), "cat_sizes": cat_sizes, "num_classes": num_classes}, model_path)
    print("[SAVE]", model_path)

    # Save encoders
    enc_path = os.path.join(args.outdir, "encoders.json")
    payload = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target_col": target_col,
        "encoders": {c: encoders[c].to_jsonable() for c in cat_cols},
        "target_encoder": y_enc.to_jsonable(),
    }
    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[SAVE]", enc_path)

    # Print top 10 most common predicted classes on a tiny sample (quick sanity)
    sample = val_df.sample(n=min(5, len(val_df)), random_state=args.seed).reset_index(drop=True)
    Xs_cat = np.column_stack([encoders[c].transform(sample[c]) for c in cat_cols])
    Xs_num = sample[num_cols].to_numpy(dtype=np.float32)
    Xs_num[:, 0] /= 3.0
    Xs_num[:, 1] /= 2.0

    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(Xs_cat, dtype=torch.long).to(device),
                       torch.as_tensor(Xs_num, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        topk = np.argsort(-probs, axis=1)[:, :5]

    print("\n[SAMPLE PREDICTIONS] (top-5 next_pitch_type)")
    for i in range(len(sample)):
        true_label = sample[target_col].iloc[i]
        preds = [(y_enc.classes_[j], float(probs[i, j])) for j in topk[i]]
        print(f"  example {i} true={true_label}  preds={preds}")


if __name__ == "__main__":
    main()
