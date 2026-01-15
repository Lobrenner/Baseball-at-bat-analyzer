# src/recommend_next.py
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
# Encoder (rebuild from encoders.json)
# -----------------------------
@dataclass
class SimpleEncoder:
    classes: List[str]
    to_idx: Dict[str, int]

    @classmethod
    def from_classes(cls, classes: List[str]) -> "SimpleEncoder":
        return cls(classes=classes, to_idx={c: i for i, c in enumerate(classes)})

    def encode(self, value: str) -> int:
        # Map unseen values to __UNK__ if present
        if value in self.to_idx:
            return self.to_idx[value]
        if "__UNK__" in self.to_idx:
            return self.to_idx["__UNK__"]
        # fallback (shouldn't happen)
        return 0


def load_model_and_encoders(modeldir: str, device: torch.device):
    model_path = os.path.join(modeldir, "model.pt")
    enc_path = os.path.join(modeldir, "encoders.json")

    ckpt = torch.load(model_path, map_location=device)
    with open(enc_path, "r", encoding="utf-8") as f:
        enc_payload = json.load(f)

    cat_cols: List[str] = enc_payload["cat_cols"]
    num_cols: List[str] = enc_payload["num_cols"]
    target_col: str = enc_payload["target_col"]

    encoders: Dict[str, SimpleEncoder] = {}
    for c in cat_cols:
        encoders[c] = SimpleEncoder.from_classes(enc_payload["encoders"][c]["classes"])

    target_enc = SimpleEncoder.from_classes(enc_payload["target_encoder"]["classes"])

    model = OutcomeMLP(
        cat_sizes=ckpt["cat_sizes"],
        num_numeric=len(num_cols),
        num_classes=ckpt["num_classes"],
        emb_dim=16,
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, encoders, target_enc, cat_cols, num_cols, target_col


def outcome_prob_map(probs: np.ndarray, target_enc: SimpleEncoder) -> Dict[str, float]:
    return {target_enc.classes[i]: float(probs[i]) for i in range(len(probs))}


def score_pitch(outcome_probs: Dict[str, float]) -> float:
    """
    K-focused objective:
    - reward K and BIP_OUT
    - penalize HIT and BB
    """
    pK = outcome_probs.get("K", 0.0)
    pOut = outcome_probs.get("BIP_OUT", 0.0)
    pBB = outcome_probs.get("BB", 0.0)
    pH = outcome_probs.get("HIT", 0.0)

    # Weights are tunable; these are reasonable v0 defaults.
    return (1.0 * pK) + (0.35 * pOut) - (0.70 * pBB) - (1.20 * pH)


def main():
    parser = argparse.ArgumentParser(description="PitchSense: recommend next pitch types for a pitcher (MLBAM id).")
    parser.add_argument("--modeldir", default="models/pitchsense_outcomes_pitcher_v1", help="Folder containing model.pt and encoders.json")
    parser.add_argument("--pitcher", required=True, type=int, help="Pitcher MLBAM id (e.g., 543037)")
    parser.add_argument("--stand", required=True, choices=["L", "R"], help="Batter handedness (L/R)")
    parser.add_argument("--p_throws", required=True, choices=["L", "R"], help="Pitcher throws (L/R)")
    parser.add_argument("--balls", required=True, type=int, help="Balls (0-3)")
    parser.add_argument("--strikes", required=True, type=int, help="Strikes (0-2)")
    parser.add_argument("--prev1", default="NONE", help="Previous pitch type (e.g., FF, SL) or NONE")
    parser.add_argument("--prev2", default="NONE", help="Two pitches ago (e.g., FF, SL) or NONE")
    parser.add_argument("--topk", type=int, default=5, help="How many recommendations to show")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoders, target_enc, cat_cols, num_cols, target_col = load_model_and_encoders(args.modeldir, device)

    # ---- Key idea: build a single "state" that we reuse for each candidate pitch_type
    base_features = {
        "pitcher": str(args.pitcher),
        "stand": args.stand,
        "p_throws": args.p_throws,
        "prev_pitch_type_1": args.prev1,
        "prev_pitch_type_2": args.prev2,
        # pitch_type will be filled in during enumeration
    }

    # Numeric features (scaled exactly like training)
    balls = float(args.balls) / 3.0
    strikes = float(args.strikes) / 2.0
    x_num = torch.tensor([[balls, strikes]], dtype=torch.float32, device=device)

    # Candidate pitch types: all known pitch types from the encoder (skip __UNK__)
    pitch_encoder = encoders["pitch_type"]
    candidates = [p for p in pitch_encoder.classes if p != "__UNK__"]

    results: List[Tuple[str, float, Dict[str, float]]] = []

    for pitch in candidates:
        # ---- Key idea: treating pitch_type as an "action" we choose
        feat = dict(base_features)
        feat["pitch_type"] = pitch

        # Encode categorical columns in the trained column order
        x_cat = []
        for c in cat_cols:
            x_cat.append(encoders[c].encode(feat[c]))
        x_cat_t = torch.tensor([x_cat], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(x_cat_t, x_num)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        probs_map = outcome_prob_map(probs, target_enc)
        s = score_pitch(probs_map)
        results.append((pitch, s, probs_map))

    # Sort by score (descending)
    results.sort(key=lambda t: t[1], reverse=True)

    print("\nPitchSense — Next Pitch Recommendations")
    print(f"  pitcher={args.pitcher}  stand={args.stand}  p_throws={args.p_throws}  count={args.balls}-{args.strikes}")
    print(f"  prev1={args.prev1}  prev2={args.prev2}")
    print(f"  modeldir={args.modeldir}  device={device}\n")

    # Helpful note if pitcher is unseen
    pitcher_idx = encoders["pitcher"].encode(str(args.pitcher))
    if "__UNK__" in encoders["pitcher"].to_idx and pitcher_idx == encoders["pitcher"].to_idx["__UNK__"]:
        print("[NOTE] This pitcher id was not seen in training; using __UNK__ embedding (more league-average behavior).\n")

    topk = max(1, min(args.topk, len(results)))
    for i in range(topk):
        pitch, s, pm = results[i]
        # Show the outcomes that matter most for your objective
        print(
            f"{i+1:>2}. pitch={pitch:<4}  score={s:+.4f}   "
            f"P(K)={pm.get('K',0):.3f}  P(BIP_OUT)={pm.get('BIP_OUT',0):.3f}  "
            f"P(BB)={pm.get('BB',0):.3f}  P(HIT)={pm.get('HIT',0):.3f}"
        )


if __name__ == "__main__":
    main()
