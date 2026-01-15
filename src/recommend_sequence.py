# src/recommend_sequence.py
from __future__ import annotations

import argparse
import copy
from typing import Dict, List, Tuple

import torch

# We reuse recommend_next logic by importing functions/classes
from recommend_next import (
    load_model_and_encoders,
    outcome_prob_map,
    score_pitch,
)


class SequenceState:
    """
    Represents the state after choosing a sequence of pitches.
    In this "contingency plan" version, balls/strikes stay fixed;
    we only update pitch history (prev1/prev2) and accumulate score.
    """
    def __init__(
        self,
        balls: int,
        strikes: int,
        prev1: str,
        prev2: str,
        score: float,
        sequence: List[str],
        step_details: List[dict] | None = None,
    ):
        self.balls = balls
        self.strikes = strikes
        self.prev1 = prev1
        self.prev2 = prev2
        self.score = score
        self.sequence = sequence
        self.step_details = step_details or []


def approximate_next_count(balls: int, strikes: int, outcome_probs: Dict[str, float]) -> Tuple[int, int]:
    """
    Approximate how the count changes after a pitch.
    We use expected tendencies instead of sampling.
    """
    p_ball = outcome_probs.get("BALL", 0.0)
    p_strike = (
        outcome_probs.get("STRIKE", 0.0)
        + outcome_probs.get("FOUL", 0.0)
    )

    # If more likely a strike than a ball, assume strike
    if p_strike >= p_ball:
        strikes = min(2, strikes + 1)
    else:
        balls = min(3, balls + 1)

    return balls, strikes

import pandas as pd

def pitcher_arsenal(data_path: str, pitcher_id: int) -> List[str]:
    df = pd.read_parquet(data_path, columns=["pitcher", "pitch_type"])
    p = str(pitcher_id)
    pitches = (
        df[df["pitcher"].astype(str) == p]["pitch_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # Drop weird ones you never want to recommend
    banned = {"PO"}  # pitchout
    pitches = [x for x in pitches if x not in banned]
    print(f"[ARSENAL] pitcher={pitcher_id} unique_pitches={len(pitches)} from {data_path}")
    return sorted(pitches)


def beam_search(
    model,
    encoders,
    target_enc,
    cat_cols,
    device,
    base_features: Dict[str, str],
    balls: int,
    strikes: int,
    prev1: str,
    prev2: str,
    beam_width: int,
    depth: int,
    data_path: str,
) -> List[SequenceState]:
    """
    Core beam search loop.
    """
    candidates = pitcher_arsenal(data_path, int(base_features["pitcher"]))
    if not candidates:
        # fallback: allow all pitch types the model knows (except __UNK__)
        pitch_encoder = encoders["pitch_type"]
        candidates = [p for p in pitch_encoder.classes if p != "__UNK__"]
        print("[WARN] Pitcher not found in data (or no pitches). Falling back to all pitch types.")


    # Initial beam (empty sequence)
    beam: List[SequenceState] = [
        SequenceState(balls, strikes, prev1, prev2, 0.0, [])
    ]

    for step in range(depth):
        new_beam: List[SequenceState] = []

        for state in beam:
            for pitch in candidates:
                feat = dict(base_features)
                feat["pitch_type"] = pitch
                feat["prev_pitch_type_1"] = state.prev1
                feat["prev_pitch_type_2"] = state.prev2

                # Encode categoricals
                x_cat = [
                    encoders[c].encode(feat[c]) for c in cat_cols
                ]
                x_cat_t = torch.tensor([x_cat], dtype=torch.long, device=device)

                # Numeric features (scaled)
                x_num = torch.tensor(
                    [[state.balls / 3.0, state.strikes / 2.0]],
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    logits = model(x_cat_t, x_num)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                probs_map = outcome_prob_map(probs, target_enc)
                pitch_score = score_pitch(probs_map)
                # Repeat penalty: allow 2 in a row sometimes, discourage 3+ strongly.
                ahead = (2 - state.strikes) + state.balls  # bigger when you’re behind / risky
                scale = 1.0 + 0.25 * ahead                 # more penalty when you can't afford it

                if pitch == state.prev1:
                    pitch_score -= 0.03 * scale
                if pitch == state.prev1 and pitch == state.prev2:
                    pitch_score -= 0.15 * scale

                detail = {
                    "pitch": pitch,
                    "step_score": float(pitch_score),
                    "P(K)": float(probs_map.get("K", 0.0)),
                    "P(BIP_OUT)": float(probs_map.get("BIP_OUT", 0.0)),
                    "P(BB)": float(probs_map.get("BB", 0.0)),
                    "P(HIT)": float(probs_map.get("HIT", 0.0)),
                }

                # Contingency planning v1: keep count fixed.
                # The user can update balls/strikes after each real pitch if they want.
                next_balls, next_strikes = state.balls, state.strikes


                new_state = SequenceState(
                    balls=next_balls,
                    strikes=next_strikes,
                    prev1=pitch,
                    prev2=state.prev1,
                    score=state.score + pitch_score,
                    sequence=state.sequence + [pitch],
                    step_details=state.step_details + [detail],
                )


                new_beam.append(new_state)

        # Keep only top-K sequences
        new_beam.sort(key=lambda s: s.score, reverse=True)
        beam = new_beam[:beam_width]

    return beam


def main():
    parser = argparse.ArgumentParser(description="PitchSense: recommend pitch sequences via beam search.")
    parser.add_argument("--modeldir", required=True)
    parser.add_argument("--pitcher", required=True, type=int)
    parser.add_argument("--stand", required=True, choices=["L", "R"])
    parser.add_argument("--p_throws", required=True, choices=["L", "R"])
    parser.add_argument("--balls", required=True, type=int)
    parser.add_argument("--strikes", required=True, type=int)
    parser.add_argument("--prev1", default="NONE")
    parser.add_argument("--prev2", default="NONE")
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--data", default="data/processed/pitchsense_outcomes_pitcher_v1.parquet", help="Parquet used to determine which pitch types a pitcher actually throws")
    parser.add_argument("--explain", action="store_true", help="Print per-step probabilities for top sequences")
    parser.add_argument("--explain_top", type=int, default=1, help="How many top sequences to explain when --explain is set")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, encoders, target_enc, cat_cols, _, _ = load_model_and_encoders(
        args.modeldir, device
    )

    base_features = {
        "pitcher": str(args.pitcher),
        "stand": args.stand,
        "p_throws": args.p_throws,
    }

    sequences = beam_search(
        model=model,
        encoders=encoders,
        target_enc=target_enc,
        cat_cols=cat_cols,
        device=device,
        base_features=base_features,
        balls=args.balls,
        strikes=args.strikes,
        prev1=args.prev1,
        prev2=args.prev2,
        beam_width=args.beam_width,
        depth=args.depth,
        data_path=args.data,
    )

    print("\nPitchSense — Recommended Pitch Sequences\n")
    for i, s in enumerate(sequences):
        print(
            f"{i+1:>2}. sequence={' → '.join(s.sequence)} "
            f"count={args.balls}-{args.strikes} "
            f"score={s.score:+.4f}"
        )

    if args.explain and sequences:
        n = max(1, min(args.explain_top, len(sequences)))
        print("\n--- Explanation (top sequences) ---")
        for i in range(n):
            s = sequences[i]
            print(f"\n#{i+1} sequence={' → '.join(s.sequence)}  total_score={s.score:+.4f}")
            for j, d in enumerate(s.step_details, start=1):
                print(
                    f"  step {j}: pitch={d['pitch']:<4} step_score={d['step_score']:+.4f}  "
                    f"P(K)={d['P(K)']:.3f}  P(BIP_OUT)={d['P(BIP_OUT)']:.3f}  "
                    f"P(BB)={d['P(BB)']:.3f}  P(HIT)={d['P(HIT)']:.3f}"
                )


if __name__ == "__main__":
    main()
