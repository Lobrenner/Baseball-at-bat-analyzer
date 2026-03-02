from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# Model (must match train_outcomes.py)
# -----------------------------
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


# -----------------------------
# Simple label encoder matching your encoders.json
# -----------------------------
@dataclass
class LabelEncoder:
    classes_: List[str]
    to_idx: Dict[str, int]

    @classmethod
    def from_classes(cls, classes: List[str]) -> "LabelEncoder":
        return cls(classes_=classes, to_idx={c: i for i, c in enumerate(classes)})

    def encode_one(self, v: str) -> int:
        v = str(v)
        if v in self.to_idx:
            return self.to_idx[v]
        # back off to __UNK__ if present
        if "__UNK__" in self.to_idx:
            return self.to_idx["__UNK__"]
        return 0


def load_model(modeldir: str, device: torch.device):
    model_path = os.path.join(modeldir, "model.pt")
    enc_path = os.path.join(modeldir, "encoders.json")

    ckpt = torch.load(model_path, map_location=device)
    with open(enc_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cat_cols = payload["cat_cols"]
    num_cols = payload["num_cols"]
    target_col = payload["target_col"]

    encoders: Dict[str, LabelEncoder] = {}
    for c in cat_cols:
        encoders[c] = LabelEncoder.from_classes(payload["encoders"][c]["classes"])

    y_enc = LabelEncoder.from_classes(payload["target_encoder"]["classes"])

    model = OutcomeMLP(
        cat_sizes=ckpt["cat_sizes"],
        num_numeric=len(num_cols),
        num_classes=ckpt["num_classes"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, encoders, y_enc, cat_cols, num_cols, target_col


def normalize_prev_action(s: str) -> str:
    """
    Accept:
      - "FF|Z1" (preferred)
      - "FF" -> coerced to "FF|OZ"
      - "NONE" stays "NONE"
    """
    s = (s or "").strip()
    if not s or s.upper() == "NONE":
        return "NONE"
    if "|" in s:
        return s
    return f"{s}|OZ"


def split_action(action: str) -> Tuple[str, str]:
    if action == "NONE":
        return ("NONE", "")
    if "|" in action:
        a, b = action.split("|", 1)
        return a, b
    return (action, "")


def probs_to_dict(probs: np.ndarray, y_enc: LabelEncoder) -> Dict[str, float]:
    return {y_enc.classes_[i]: float(probs[i]) for i in range(len(probs))}


def score_objective(pm: Dict[str, float]) -> float:
    """
    Simple default objective:
      + K
      + small reward for BIP_OUT
      - BB
      - HIT
    """
    return (
        1.0 * pm.get("K", 0.0)
        + 0.35 * pm.get("BIP_OUT", 0.0)
        - 0.70 * pm.get("BB", 0.0)
        - 1.20 * pm.get("HIT", 0.0)
    )


def main():
    parser = argparse.ArgumentParser(description="Recommend next pitch_action (pitch_type|loc_bucket).")
    parser.add_argument("--modeldir", default="models/pitchsense_outcomes_v0")
    parser.add_argument("--pitcher", required=True, type=int)
    parser.add_argument("--stand", required=True, choices=["L", "R"])
    parser.add_argument("--p_throws", required=True, choices=["L", "R"])
    parser.add_argument("--balls", required=True, type=int)
    parser.add_argument("--strikes", required=True, type=int)
    parser.add_argument("--prev1", default="NONE", help='e.g. "FF|Z1" or "FF" or NONE')
    parser.add_argument("--prev2", default="NONE", help='e.g. "SL|OZ" or "SL" or NONE')
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print JSON output only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoders, y_enc, cat_cols, num_cols, target_col = load_model(args.modeldir, device)

    # Safety check: ensure this model supports pitch_action
    if "pitch_action" not in encoders:
        raise KeyError(
            "encoders.json is missing 'pitch_action'. "
            "Did you retrain after updating prep_outcomes.py/train_outcomes.py?"
        )

    prev1 = normalize_prev_action(args.prev1)
    prev2 = normalize_prev_action(args.prev2)

    # Numeric features (must match training scaling)
    x_num = np.array([[args.balls / 3.0, args.strikes / 2.0]], dtype=np.float32)
    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)

    # Candidate actions: all pitch_action classes (exclude __UNK__)
    action_encoder = encoders["pitch_action"]
    candidates = [a for a in action_encoder.classes_ if a != "__UNK__"]

    results = []

    for action in candidates:
        feat = {
            "pitcher": str(args.pitcher),
            "stand": args.stand,
            "p_throws": args.p_throws,
            "prev_action_1": prev1,
            "prev_action_2": prev2,
            "pitch_action": action,
        }

        # encode cats in trained order
        x_cat = [encoders[c].encode_one(feat[c]) for c in cat_cols]
        x_cat_t = torch.tensor([x_cat], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(x_cat_t, x_num_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pm = probs_to_dict(probs, y_enc)
        s = score_objective(pm)

        results.append((action, s, pm))

    results.sort(key=lambda t: t[1], reverse=True)
    topk = max(1, min(args.topk, len(results)))

    out = {
        "mode": "next_pitch",
        "state": {
            "pitcher": args.pitcher,
            "stand": args.stand,
            "p_throws": args.p_throws,
            "balls": args.balls,
            "strikes": args.strikes,
            "prev_action_1": prev1,
            "prev_action_2": prev2,
        },
        "recommendations": [],
    }

    for i in range(topk):
        action, score, pm = results[i]
        pitch_type, loc_bucket = split_action(action)

        out["recommendations"].append(
            {
                "rank": i + 1,
                "pitch_action": action,
                "pitch_type": pitch_type,
                "loc_bucket": loc_bucket,
                "score": score,
                "why": {
                    "P(K)": pm.get("K", 0.0),
                    "P(BB)": pm.get("BB", 0.0),
                    "P(HIT)": pm.get("HIT", 0.0),
                    "P(BIP_OUT)": pm.get("BIP_OUT", 0.0),
                },
                "probs": pm,  # full distribution (optional but useful)
            }
        )

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print("\nPitchSense — Next Pitch+Location Recommendations\n")
    print(
        f"pitcher={args.pitcher} stand={args.stand} p_throws={args.p_throws} "
        f"count={args.balls}-{args.strikes} prev1={prev1} prev2={prev2}\n"
    )

    for rec in out["recommendations"]:
        print(
            f"{rec['rank']:>2}. {rec['pitch_action']:<8} "
            f"(pitch={rec['pitch_type']:<3} loc={rec['loc_bucket']:<2}) "
            f"score={rec['score']:+.4f}  "
            f"P(K)={rec['why']['P(K)']:.3f} P(BIP_OUT)={rec['why']['P(BIP_OUT)']:.3f} "
            f"P(BB)={rec['why']['P(BB)']:.3f} P(HIT)={rec['why']['P(HIT)']:.3f}"
        )


if __name__ == "__main__":
    main()
