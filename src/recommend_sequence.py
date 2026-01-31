from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


## EXAMPLE EXECUTE COMMAND: python src\pybaseball_test.py --modeldir models\Test_v5 --data data\Test\big_data.parquet --pitcher 669373 --stand L --p_throws L --balls 3 --strikes 0 --beam_width 2 --depth 4 --batter 701538 --explain  --explain_top 2
## For the above command, change the names of files/folders to whatever you have yours for --modeldir and --data



# Model (must match train_outcomes.py)
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


# -----------------------------
# Arsenal (restored from original)
# -----------------------------
def pitcher_arsenal(data_path: str, pitcher_id: int) -> List[str]:
    df = pd.read_parquet(data_path, columns=["pitcher", "pitch_action"])
    p = str(pitcher_id)

    sub = df[df["pitcher"].astype(str) == p]["pitch_action"].dropna().astype(str)

    # pitch_action looks like "FF|Z1" -> pitch_type is before "|"
    pitches = sub.str.split("|", n=1, expand=True)[0].unique().tolist()

    banned = {"PO"}
    pitches = [x for x in pitches if x not in banned]
    print(f"[ARSENAL] pitcher={pitcher_id} unique_pitches={len(pitches)} from {data_path}")
    return sorted(pitches)



def filter_actions_by_arsenal(actions: List[str], allowed_pitches: List[str]) -> List[str]:
    allowed = set(allowed_pitches)
    out: List[str] = []
    for a in actions:
        if a == "__UNK__":
            continue
        pitch = a.split("|", 1)[0]
        if pitch in allowed:
            out.append(a)
    return out


# -----------------------------
# Helpers
# -----------------------------
TERMINAL = {"K", "BB", "HIT", "BIP_OUT"}


def normalize_prev_action(s: str) -> str:
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


def score_objective(pm: Dict[str, float], balls: int, strikes: int) -> float:
    """
    make a score for each pitch chosen, can be adjusted later
    scores are relative only to other pitches in sequence
    """
    base = (
        1.0 * pm.get("K", 0.0)
        + 0.35 * pm.get("BIP_OUT", 0.0)
        - 0.70 * pm.get("BB", 0.0)
        - 1.20 * pm.get("HIT", 0.0)
    )

    p_ball = pm.get("BALL", 0.0)
    p_strike = pm.get("STRIKE", 0.0)
    p_foul = pm.get("FOUL", 0.0)

    # Strikes are more valuable early to get ahead in the count
    if strikes == 0:
        strike_w = 0.10
    elif strikes == 1:
        strike_w = 0.06
    else:  # strikes == 2
        strike_w = 0.02

    # Fouls only add strike value when strikes < 2
    foul_w = 0.06 if strikes < 2 else 0.0

    # Balls are mildly bad; more bad when you're already at 2+ balls
    ball_w = 0.05 if balls < 2 else 0.10

    shaped = base + strike_w * p_strike + foul_w * p_foul - ball_w * p_ball
    return shaped



def terminal_probability(pm: Dict[str, float]) -> float:
    return sum(pm.get(k, 0.0) for k in TERMINAL)


def clamp_count(b: int, s: int) -> Tuple[int, int]:
    return max(0, min(3, b)), max(0, min(2, s))


def transition_count(b: int, s: int, outcome: str) -> Tuple[int, int]:
    if outcome == "BALL":
        return clamp_count(b + 1, s)
    if outcome == "STRIKE":
        return clamp_count(b, s + 1)
    if outcome == "FOUL":
        return clamp_count(b, s if s >= 2 else s + 1)
    return b, s


def normalize_dist(dist: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    tot = sum(dist.values())
    if tot <= 0:
        return dist
    return {k: v / tot for k, v in dist.items()}


def advance_count_distribution( #Asked chatgpt to do this and it kinda sucked ngl
    dist: Dict[Tuple[int, int], float],
    pm: Dict[str, float],
) -> Dict[Tuple[int, int], float]:
    """
    Propagate count distribution using BALL/STRIKE/FOUL, and treat everything else as "stay".
    This keeps counts integer and realistic.
    """
    p_ball = pm.get("BALL", 0.0)
    p_strike = pm.get("STRIKE", 0.0)
    p_foul = pm.get("FOUL", 0.0)

    p_move = p_ball + p_strike + p_foul
    p_stay = max(0.0, 1.0 - p_move)

    new_dist: Dict[Tuple[int, int], float] = {}

    for (b, s), w in dist.items():
        # stay
        new_dist[(b, s)] = new_dist.get((b, s), 0.0) + w * p_stay

        # move
        nb, ns = transition_count(b, s, "BALL")
        new_dist[(nb, ns)] = new_dist.get((nb, ns), 0.0) + w * p_ball

        nb, ns = transition_count(b, s, "STRIKE")
        new_dist[(nb, ns)] = new_dist.get((nb, ns), 0.0) + w * p_strike

        nb, ns = transition_count(b, s, "FOUL")
        new_dist[(nb, ns)] = new_dist.get((nb, ns), 0.0) + w * p_foul

    return normalize_dist(new_dist)


def expected_count(dist: Dict[Tuple[int, int], float]) -> Tuple[float, float]:
    eb = sum(b * p for (b, s), p in dist.items())
    es = sum(s * p for (b, s), p in dist.items())
    return eb, es


def count_risk_penalty_from_dist(dist: Dict[Tuple[int, int], float]) -> float:
    # same spirit as earlier: avoid drifting into 3-ball danger
    p3 = sum(p for (b, s), p in dist.items() if b == 3)
    p2 = sum(p for (b, s), p in dist.items() if b == 2)
    p2str = sum(p for (b, s), p in dist.items() if s == 2)
    return 0.06 * p3 + 0.02 * p2 - 0.01 * p2str


def pitch_type_of(action_or_prev: str) -> str:
    if action_or_prev == "NONE":
        return "NONE"
    return action_or_prev.split("|", 1)[0]


def repeat_penalty_original_style( #asked chat to do this to and it sucked again, but whatever ill clean it up later
    prev1: str,
    prev2: str,
    action: str,
    count_dist: Dict[Tuple[int, int], float],
) -> float:
    """
    Port of your original logic:
      - allow 2 in a row sometimes
      - discourage 3+ strongly
      - scale up penalty when you're "behind / risky"
    Applied to pitch TYPE (not location-specific), which is more baseball-realistic.
    """
    pitch = pitch_type_of(action)
    p1 = pitch_type_of(prev1)
    p2 = pitch_type_of(prev2)

    eb, es = expected_count(count_dist)
    ahead = (2 - es) + eb  # bigger when behind/risky
    scale = 1.0 + 0.25 * ahead

    pen = 0.0
    if pitch == p1:
        pen += 0.03 * scale
    if pitch == p1 and pitch == p2:
        pen += 0.15 * scale
    return pen

def alternation_penalty(
    prev1: str,
    prev2: str,
    action: str,
    count_dist: Dict[Tuple[int, int], float],
) -> float:
    """
    Stop sequence form alternating between the two best strike pitches
    Hopefully when the model and datasets are perfected it won't be necessary to penalize anymore
    """
    a = pitch_type_of(action)
    p1 = pitch_type_of(prev1)
    p2 = pitch_type_of(prev2)

    # Detect alternating sequence
    if p2 != "NONE" and a == p2 and a != p1:
        _, es = expected_count(count_dist)
        scale = 1.0 + 1.0 * min(es, 2.0)   # 0 strikes -> 1.0, 2 strikes -> 3.0
        return 0.06 * scale                # base penalty
    return 0.0



def model_predict(
    model: nn.Module,
    encoders: Dict[str, LabelEncoder],
    y_enc: LabelEncoder,
    cat_cols: List[str],
    balls: int,
    strikes: int,
    feat: Dict[str, str],
    device: torch.device,
) -> Dict[str, float]:
    # numeric scaling matches training
    x_num = np.array([[balls / 3.0, strikes / 2.0]], dtype=np.float32)
    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)

    x_cat = [encoders[c].encode_one(feat[c]) for c in cat_cols]
    x_cat_t = torch.tensor([x_cat], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(x_cat_t, x_num_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs_to_dict(probs, y_enc)


# Beam search with stochastic counts

GAMMA = 0.90

@dataclass
class BeamState:
    count_dist: Dict[Tuple[int, int], float]  # {(balls,strikes): prob}
    prev1: str
    prev2: str
    seq: List[Dict[str, Any]]
    score: float
    done: bool


def beam_search(
    model: nn.Module,
    encoders: Dict[str, LabelEncoder],
    y_enc: LabelEncoder,
    cat_cols: List[str],
    pitcher: int,
    stand: str,
    p_throws: str,
    balls: int,
    strikes: int,
    prev1: str,
    prev2: str,
    depth: int,
    beam_width: int,
    device: torch.device,
    data_path: Optional[str] = None,
) -> List[BeamState]:
    if "pitch_action" not in encoders:
        raise KeyError("Missing 'pitch_action' in encoders; retrain with pitch_action.")

    action_encoder = encoders["pitch_action"]
    actions = [a for a in action_encoder.classes_ if a != "__UNK__"]

    # Restore arsenal restriction (optional but recommended)
    if data_path is not None:
        try:
            allowed = pitcher_arsenal(data_path, pitcher)
            if allowed:
                actions = filter_actions_by_arsenal(actions, allowed)
        except Exception as e:
            print(f"[ARSENAL WARNING] could not load arsenal from {data_path}: {e}")
            print("[ARSENAL WARNING] falling back to all actions from encoder.")

    if not actions:
        raise RuntimeError("No candidate actions available after filtering.")

    init = BeamState(
        count_dist={(balls, strikes): 1.0},
        prev1=prev1,
        prev2=prev2,
        seq=[],
        score=0.0,
        done=False,
    )
    beams = [init]

    for step in range(1, depth + 1):
        new_beams: List[BeamState] = []

        for b in beams:
            if b.done:
                new_beams.append(b)
                continue

            for action in actions:
                step_score_exp = 0.0
                next_dist_accum: Dict[Tuple[int, int], float] = {}
                p_term_exp = 0.0

                # expected over current count distribution
                for (cb, cs), w in b.count_dist.items():
                    feat = {
                        "pitcher": str(pitcher),
                        "stand": stand,
                        "p_throws": p_throws,
                        "prev_action_1": b.prev1,
                        "prev_action_2": b.prev2,
                        "pitch_action": action,
                    }

                    pm = model_predict(
                        model=model,
                        encoders=encoders,
                        y_enc=y_enc,
                        cat_cols=cat_cols,
                        balls=cb,
                        strikes=cs,
                        feat=feat,
                        device=device,
                    )

                    step_score_exp += w * score_objective(pm, cb, cs)
                    p_term_exp += w * terminal_probability(pm)

                    nd = advance_count_distribution({(cb, cs): 1.0}, pm)
                    for k, v in nd.items():
                        next_dist_accum[k] = next_dist_accum.get(k, 0.0) + w * v

                next_dist = normalize_dist(next_dist_accum)

                # Restore original-style repeat penalty (scaled by risky counts)
                step_score_exp -= repeat_penalty_original_style(b.prev1, b.prev2, action, b.count_dist)
                #ALternating pitch penalty, porevent ABABABABAB
                step_score_exp -= alternation_penalty(b.prev1, b.prev2, action, b.count_dist)
                # walk penalty
                step_score_exp -= count_risk_penalty_from_dist(next_dist)

                


                # Early stop if terminal is likely (keep, but can be tuned)
                done = b.done
                if p_term_exp >= 0.60:
                    done = False     #This makes it so early stop never happens, still deciding if I want that or not

                pitch_type, loc_bucket = split_action(action)
                eb0, es0 = expected_count(b.count_dist)
                eb1, es1 = expected_count(next_dist)

                # "Why" fields: show probs at MAP count (most likely count state)
                (map_b, map_s) = max(b.count_dist.items(), key=lambda kv: kv[1])[0]
                feat_map = {
                    "pitcher": str(pitcher),
                    "stand": stand,
                    "p_throws": p_throws,
                    "prev_action_1": b.prev1,
                    "prev_action_2": b.prev2,
                    "pitch_action": action,
                }
                pm_map = model_predict(
                    model=model,
                    encoders=encoders,
                    y_enc=y_enc,
                    cat_cols=cat_cols,
                    balls=map_b,
                    strikes=map_s,
                    feat=feat_map,
                    device=device,
                )

                step_obj = {
                    "step": step,
                    "pitch_action": action,
                    "pitch_type": pitch_type,
                    "loc_bucket": loc_bucket,
                    "count_exp_before": [round(eb0, 3), round(es0, 3)],
                    "count_exp_after": [round(eb1, 3), round(es1, 3)],
                    "count_dist_before": {f"{k[0]}-{k[1]}": round(v, 4) for k, v in sorted(b.count_dist.items())},
                    "count_dist_after": {f"{k[0]}-{k[1]}": round(v, 4) for k, v in sorted(next_dist.items())},
                    "p_terminal_exp": round(p_term_exp, 4),
                    "why": {
                        "MAP_count": f"{map_b}-{map_s}",
                        "P(K)": pm_map.get("K", 0.0),
                        "P(BIP_OUT)": pm_map.get("BIP_OUT", 0.0),
                        "P(BB)": pm_map.get("BB", 0.0),
                        "P(HIT)": pm_map.get("HIT", 0.0),
                        "P(BALL)": pm_map.get("BALL", 0.0),
                        "P(STRIKE)": pm_map.get("STRIKE", 0.0),
                        "P(FOUL)": pm_map.get("FOUL", 0.0),
                    },
                    "step_score_exp": step_score_exp,
                }

                new_state = BeamState(
                    count_dist=next_dist,
                    prev1=action,
                    prev2=b.prev1,
                    seq=b.seq + [step_obj],
                    score=b.score + (GAMMA ** (step - 1)) * step_score_exp,
                    done=done,
                )
                new_beams.append(new_state)

        new_beams.sort(key=lambda bs: bs.score, reverse=True)
        beams = new_beams[:beam_width]

    return beams


def main():
    parser = argparse.ArgumentParser(
        description="Recommend a pitch_action sequence using beam search (stochastic count distribution)."
    )
    parser.add_argument("--modeldir", default="models/pitchsense_outcomes_v0")
    parser.add_argument("--data", default=None, help="Optional parquet path used to infer pitcher arsenal")
    parser.add_argument("--pitcher", required=True, type=int)
    parser.add_argument("--stand", required=True, choices=["L", "R"])
    parser.add_argument("--p_throws", required=True, choices=["L", "R"])
    parser.add_argument("--balls", required=True, type=int)
    parser.add_argument("--strikes", required=True, type=int)
    parser.add_argument("--prev1", default="NONE")
    parser.add_argument("--prev2", default="NONE")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    #parser.add_argument("--gamma", type=float, default=0.90)


    # Restored explain flags
    parser.add_argument("--explain", action="store_true", help="Print extra per-step probability explanations")
    parser.add_argument("--explain_top", type=int, default=3, help="How many top sequences to explain")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoders, y_enc, cat_cols, num_cols, target_col = load_model(args.modeldir, device)

    prev1 = normalize_prev_action(args.prev1)
    prev2 = normalize_prev_action(args.prev2)

    beams = beam_search(
        model=model,
        encoders=encoders,
        y_enc=y_enc,
        cat_cols=cat_cols,
        pitcher=args.pitcher,
        stand=args.stand,
        p_throws=args.p_throws,
        balls=args.balls,
        strikes=args.strikes,
        prev1=prev1,
        prev2=prev2,
        depth=args.depth,
        beam_width=args.beam_width,
        device=device,
        data_path=args.data,
    )

    out = {
        "mode": "sequence",
        "state": {
            "pitcher": args.pitcher,
            "stand": args.stand,
            "p_throws": args.p_throws,
            "balls": args.balls,
            "strikes": args.strikes,
            "prev_action_1": prev1,
            "prev_action_2": prev2,
        },
        "params": {"depth": args.depth, "beam_width": args.beam_width},
        "sequences": [],
    }

    topk = max(1, min(args.topk, len(beams)))
    for i in range(topk):
        b = beams[i]
        eb, es = expected_count(b.count_dist)
        out["sequences"].append(
            {
                "rank": i + 1,
                "total_score": b.score,
                "final_count_dist": {f"{k[0]}-{k[1]}": round(v, 4) for k, v in sorted(b.count_dist.items())},
                "final_count_exp": [round(eb, 3), round(es, 3)],
                "done": b.done,
                "steps": b.seq,
            }
        )

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print("\nPitchSense — Sequence Recommendations (stochastic count)\n")
    print(
        f"pitcher={args.pitcher} stand={args.stand} p_throws={args.p_throws} "
        f"start_count={args.balls}-{args.strikes} prev1={prev1} prev2={prev2}\n"
    )

    if args.data:
        print(f"[INFO] arsenal_source={args.data}\n")

    for seq in out["sequences"]:
        print(f"== Sequence {seq['rank']}  total_score={seq['total_score']:+.4f}")
        #  done={seq['done']} ==")
        #  print(f"  final_count_exp={seq['final_count_exp']}")

        # show top few end states
        items = sorted(seq["final_count_dist"].items(), key=lambda kv: kv[1], reverse=True)[:5]
        #   print("  final_count_dist_top:", ", ".join([f"{k}:{v:.3f}" for k, v in items]))

        for step in seq["steps"]:
            cb = step["count_exp_before"]
            ca = step["count_exp_after"]
            print(
                f"  step {step['step']}: {step['pitch_action']:<8} "
               #   f"(pitch={step['pitch_type']:<3} loc={step['loc_bucket']:<2})  "
               # f"exp_count {cb[0]}-{cb[1]} -> {ca[0]}-{ca[1]}  "
               # f"p_term_exp={step['p_terminal_exp']:.3f}  "
                f"score_exp={step['step_score_exp']:+.4f}"  #MAP={step['why']['MAP_count']}"
            )
        print()

    # Restored "explain" output (original-style)
    if args.explain and out["sequences"]:
        n = max(1, min(args.explain_top, len(out["sequences"])))
        print("\n--- Explanation (top sequences) ---")
        for i in range(n):
            s = out["sequences"][i]
            seq_str = " → ".join([st["pitch_action"] for st in s["steps"]])
            print(f"\n#{i+1} sequence={seq_str}  total_score={s['total_score']:+.4f}")
            for st in s["steps"]:
                w = st["why"]
                print(
                    f"  step {st['step']}: action={st['pitch_action']:<8} step_score={st['step_score_exp']:+.4f}  "
                    f"P(K)={w['P(K)']:.3f}  P(BIP_OUT)={w['P(BIP_OUT)']:.3f}  "
                    f"P(BB)={w['P(BB)']:.3f}  P(HIT)={w['P(HIT)']:.3f}"
                )

    print("\nTip: add --json for full structured count distributions.\n")


if __name__ == "__main__":
    main()
