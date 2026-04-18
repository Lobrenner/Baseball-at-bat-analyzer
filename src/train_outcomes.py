from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


class PitchOutcomeDataset(Dataset):
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray):
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class OutcomeMLP(nn.Module):
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
        self.net = nn.Sequential(
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
        return self.net(x)


def split_train_val_rows(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int(len(df) * (1 - val_frac))
    return df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[idx[cut:]].reset_index(drop=True)


def split_train_val_grouped(
    df: pd.DataFrame,
    group_cols: List[str],
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Grouped split requested, but missing columns: {missing}")

    group_keys = df[group_cols].astype(str).agg("||".join, axis=1)
    uniq = group_keys.drop_duplicates().to_numpy()

    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    cut = int(len(uniq) * (1 - val_frac))
    train_groups = set(uniq[:cut])
    val_groups = set(uniq[cut:])

    train_mask = group_keys.isin(train_groups)
    val_mask = group_keys.isin(val_groups)

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    return train_df, val_df


def batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_report_from_cm(cm: np.ndarray, class_names: List[str]) -> List[dict]:
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())

        rows.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def macro_f1_from_report(report: List[dict]) -> float:
    if not report:
        return 0.0
    real_rows = [r for r in report if r["class"] != "__UNK__"]
    if not real_rows:
        return 0.0
    return float(np.mean([r["f1"] for r in real_rows]))


def print_per_class_report(report: List[dict], focus_classes: List[str] | None = None):
    rows = report
    if focus_classes is not None:
        rows = [r for r in report if r["class"] in focus_classes]

    print("[PER-CLASS METRICS]")
    for r in rows:
        print(
            f"  {r['class']:<8} "
            f"precision={r['precision']:.3f} "
            f"recall={r['recall']:.3f} "
            f"f1={r['f1']:.3f} "
            f"support={r['support']}"
        )


def print_confusion_pairs(cm: np.ndarray, class_names: List[str], top_k: int = 8):
    pairs = []
    n = len(class_names)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            count = int(cm[i, j])
            if count > 0:
                pairs.append((count, class_names[i], class_names[j]))

    pairs.sort(reverse=True)
    print(f"[TOP CONFUSIONS] top_k={top_k}")
    for count, true_name, pred_name in pairs[:top_k]:
        print(f"  true={true_name:<8} predicted={pred_name:<8} count={count}")

def print_avg_predicted_probs_by_true_class(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    focus_pred_classes: List[str],
):
    print("[AVG PREDICTED PROBS BY TRUE CLASS]")
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    for true_idx, true_name in enumerate(class_names):
        if true_name == "__UNK__":
            continue

        mask = (y_true == true_idx)
        if mask.sum() == 0:
            continue

        parts = []
        for pred_name in focus_pred_classes:
            if pred_name not in name_to_idx:
                continue
            pred_idx = name_to_idx[pred_name]
            avg_p = float(probs[mask, pred_idx].mean())
            parts.append(f"P({pred_name})={avg_p:.4f}")

        print(f"  true={true_name:<8} n={int(mask.sum()):<6} " + "  ".join(parts))

def make_class_weights(
    y_encoded: np.ndarray,
    num_classes: int,
    mode: str = "sqrt_inv",
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> np.ndarray:
    counts = np.bincount(y_encoded, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)

    if mode == "inv":
        weights = 1.0 / counts
    elif mode == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts)
    elif mode == "log_inv":
        weights = 1.0 / np.log1p(counts)
    else:
        raise ValueError(f"Unknown class-weight mode: {mode}")

    weights = weights / weights.mean()
    weights = np.clip(weights, min_weight, max_weight)
    return weights.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train PitchSense outcome model (PyTorch multiclass).")
    parser.add_argument("--data", default="data/processed/pitchsense_outcomes_v0.parquet")
    parser.add_argument("--outdir", default="models/pitchsense_outcomes_v0")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument(
        "--split_mode",
        choices=["row", "game", "atbat"],
        default="atbat",
        help="row=random row split, game=split by game_pk, atbat=split by (game_pk, at_bat_number)",
    )

    parser.add_argument("--early_stop_patience", type=int, default=4)

    parser.add_argument("--use_class_weights", action="store_true", help="Use class weights in CrossEntropyLoss")
    parser.add_argument(
        "--class_weight_mode",
        choices=["inv", "sqrt_inv", "log_inv"],
        default="sqrt_inv",
        help="How aggressively to upweight rare classes",
    )
    parser.add_argument("--class_weight_min", type=float, default=0.25)
    parser.add_argument("--class_weight_max", type=float, default=4.0)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.data)
    print("[LOAD]", df.shape)
    print("[OUTCOME COUNTS]\n", df["outcome"].value_counts().head(12))

    cat_cols = [
        "pitcher",
        "batter",
        "stand",
        "p_throws",
        "inning_bucket",
        "base_state",
        "prev_action_1",
        "prev_action_2",
        "pitch_action",
        "tto_bucket",
    ]
    num_cols = ["balls", "strikes", "runners_count", "outs"]
    target_col = "outcome"

    if args.split_mode == "row":
        train_df, val_df = split_train_val_rows(df, val_frac=args.val_frac, seed=args.seed)
    elif args.split_mode == "game":
        train_df, val_df = split_train_val_grouped(
            df, group_cols=["game_pk"], val_frac=args.val_frac, seed=args.seed
        )
    else:
        train_df, val_df = split_train_val_grouped(
            df, group_cols=["game_pk", "at_bat_number"], val_frac=args.val_frac, seed=args.seed
        )

    print(f"[SPLIT MODE] {args.split_mode}")
    print("[SPLIT] train=", train_df.shape, "val=", val_df.shape)

    encoders: Dict[str, LabelEncoder] = {c: LabelEncoder.fit(train_df[c]) for c in cat_cols}
    y_enc = LabelEncoder.fit(train_df[target_col])

    X_train_cat = np.column_stack([encoders[c].transform(train_df[c]) for c in cat_cols])
    X_val_cat = np.column_stack([encoders[c].transform(val_df[c]) for c in cat_cols])

    X_train_num = train_df[num_cols].to_numpy(dtype=np.float32)
    X_val_num = val_df[num_cols].to_numpy(dtype=np.float32)

    X_train_num[:, 0] /= 3.0
    X_train_num[:, 1] /= 2.0
    X_train_num[:, 2] /= 3.0
    X_train_num[:, 3] /= 2.0

    X_val_num[:, 0] /= 3.0
    X_val_num[:, 1] /= 2.0
    X_val_num[:, 2] /= 3.0
    X_val_num[:, 3] /= 2.0

    y_train = y_enc.transform(train_df[target_col])
    y_val = y_enc.transform(val_df[target_col])

    class_weights = None
    if args.use_class_weights:
        class_weights_np = make_class_weights(
            y_train,
            num_classes=len(y_enc.classes_),
            mode=args.class_weight_mode,
            min_weight=args.class_weight_min,
            max_weight=args.class_weight_max,
        )
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

        print("[CLASS WEIGHTS]")
        for cls_name, w in zip(y_enc.classes_, class_weights_np):
            print(f"  {cls_name:<8} weight={w:.4f}")

    train_ds = PitchOutcomeDataset(X_train_cat, X_train_num, y_train)
    val_ds = PitchOutcomeDataset(X_val_cat, X_val_num, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[DEVICE]", device)

    cat_sizes = [len(encoders[c].classes_) for c in cat_cols]
    num_classes = len(y_enc.classes_)

    model = OutcomeMLP(
        cat_sizes=cat_sizes,
        num_numeric=len(num_cols),
        num_classes=num_classes,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_used = 0

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = tr_acc = 0.0
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
        va_loss = va_acc = 0.0
        va_batches = 0

        all_val_preds = []
        all_val_true = []
        all_val_probs = []

        with torch.no_grad():
            for Xc, Xn, yb in val_loader:
                Xc, Xn, yb = Xc.to(device), Xn.to(device), yb.to(device)
                logits = model(Xc, Xn)
                loss = loss_fn(logits, yb)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                va_loss += loss.item()
                va_acc += batch_accuracy(logits, yb)
                va_batches += 1

                all_val_preds.append(preds.cpu().numpy())
                all_val_true.append(yb.cpu().numpy())
                all_val_probs.append(probs.cpu().numpy())

        all_val_preds = np.concatenate(all_val_preds)
        all_val_true = np.concatenate(all_val_true)
        all_val_probs = np.concatenate(all_val_probs, axis=0)

        cm = confusion_matrix_np(all_val_true, all_val_preds, num_classes=num_classes)
        report = per_class_report_from_cm(cm, y_enc.classes_)
        macro_f1 = macro_f1_from_report(report)

        tr_loss_avg = tr_loss / max(tr_batches, 1)
        tr_acc_avg = tr_acc / max(tr_batches, 1)
        va_loss_avg = va_loss / max(va_batches, 1)
        va_acc_avg = va_acc / max(va_batches, 1)

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss_avg,
                "train_acc": tr_acc_avg,
                "val_loss": va_loss_avg,
                "val_acc": va_acc_avg,
                "val_macro_f1": macro_f1,
            }
        )

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={tr_loss_avg:.4f} train_acc={tr_acc_avg:.4f} | "
            f"val_loss={va_loss_avg:.4f} val_acc={va_acc_avg:.4f} val_macro_f1={macro_f1:.4f}"
        )
        print_per_class_report(report, focus_classes=["BB", "BIP_OUT", "HIT", "K"])
        print_confusion_pairs(cm, y_enc.classes_, top_k=8)
        print_avg_predicted_probs_by_true_class(
            all_val_true,
            all_val_probs,
            y_enc.classes_,
            focus_pred_classes=["HIT", "BIP_OUT", "K", "BB"],
        )
        
        if va_loss_avg < best_val_loss:
            best_val_loss = va_loss_avg
            best_epoch = epoch
            patience_used = 0
            best_state = {
                "state_dict": model.state_dict(),
                "cat_sizes": cat_sizes,
                "num_classes": num_classes,
                "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
            }
        else:
            patience_used += 1
            if patience_used >= args.early_stop_patience:
                print(f"[EARLY STOP] no val_loss improvement for {args.early_stop_patience} epochs")
                break

    print(f"[BEST] epoch={best_epoch} val_loss={best_val_loss:.4f}")

    if best_state is None:
        best_state = {
            "state_dict": model.state_dict(),
            "cat_sizes": cat_sizes,
            "num_classes": num_classes,
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }

    model_path = os.path.join(args.outdir, "model.pt")
    torch.save(best_state, model_path)
    print("[SAVE]", model_path)

    enc_path = os.path.join(args.outdir, "encoders.json")
    payload = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "target_col": target_col,
        "encoders": {c: encoders[c].to_jsonable() for c in cat_cols},
        "target_encoder": y_enc.to_jsonable(),
        "split_mode": args.split_mode,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    with open(enc_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("[SAVE]", enc_path)


if __name__ == "__main__":
    main()