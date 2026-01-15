from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

# We keep only what we need right now (state, action, labeling fields)
KEEP_COLS = [
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter",
    "pitcher",
    "stand",
    "p_throws",
    "balls",
    "strikes",
    "pitch_type",
    "description",
    "events",
]


# --- Outcome mapping helpers ---
HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}
# Common “ball in play -> out” event labels in Statcast
BIP_OUT_EVENTS = {
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "triple_play",
    "fielders_choice_out",
    "sac_fly",
    "sac_bunt",
    "sac_bunt_double_play",
    "strikeout_double_play",  # treat as K terminal too (below)
}

STRIKE_DESCRIPTIONS = {"called_strike", "swinging_strike", "swinging_strike_blocked"}
FOUL_DESCRIPTIONS = {"foul", "foul_tip", "foul_bunt"}


def load_parquets(indir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(indir, "statcast_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in: {indir}")

    dfs = []
    for p in paths:
        dfs.append(pd.read_parquet(p, columns=KEEP_COLS))
    return pd.concat(dfs, ignore_index=True)


def add_prev_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    keys = ["game_pk", "at_bat_number"]

    # Key idea: shift() creates “previous pitch” features within the same at-bat
    df["prev_pitch_type_1"] = df.groupby(keys)["pitch_type"].shift(1).fillna("NONE")
    df["prev_pitch_type_2"] = df.groupby(keys)["pitch_type"].shift(2).fillna("NONE")
    return df


def label_outcome(row: pd.Series) -> str:
    """
    Returns a label representing the outcome AFTER this pitch.
    Terminal outcomes are preferred when `events` is present.
    Otherwise we fall back to pitch-by-pitch `description`.
    """
    ev = row.get("events")
    desc = row.get("description")

    # --- Terminal outcomes from events (end-of-PA outcomes) ---
    if isinstance(ev, str) and ev:
        if ev == "strikeout" or ev == "strikeout_double_play":
            return "K"
        if ev in WALK_EVENTS:
            return "BB"
        if ev in HIT_EVENTS:
            return "HIT"
        if ev in BIP_OUT_EVENTS:
            return "BIP_OUT"
        # Other events exist, but v0 can bucket them later
        # e.g. "fielders_choice", "catcher_interf", etc.
        return "OTHER_EVENT"

    # --- Non-terminal pitch outcomes from description ---
    if desc == "ball":
        return "BALL"
    if desc in STRIKE_DESCRIPTIONS:
        return "STRIKE"
    if desc in FOUL_DESCRIPTIONS:
        return "FOUL"
    if isinstance(desc, str) and desc.startswith("hit_into_play"):
        # If events was missing, this at least tells us the ball was put in play
        return "IN_PLAY"

    return "OTHER_PITCH"


def main():
    parser = argparse.ArgumentParser(description="Prepare pitch-level outcome dataset for PitchSense.")
    parser.add_argument("--indir", default="data/raw", help="Directory containing statcast_*.parquet files")
    parser.add_argument("--out", default="data/processed/pitchsense_outcomes_v0.parquet", help="Output parquet path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = load_parquets(args.indir)
    print("[LOAD]", df.shape)

    # Clean rows that would break modeling
    df = df.dropna(subset=["pitch_type", "balls", "strikes", "stand", "p_throws"])
    print("[CLEAN]", df.shape)

    df = add_prev_pitch_features(df)

    # Key idea: apply outcome labeling per pitch
    df["outcome"] = df.apply(label_outcome, axis=1)
    df["outcome"] = df["outcome"].replace({"OTHER_PITCH": "OTHER", "OTHER_EVENT": "OTHER"})

    # Keep a modeling-friendly subset
    out_cols = [
        "pitcher",
        "stand",
        "p_throws",
        "balls",
        "strikes",
        "prev_pitch_type_1",
        "prev_pitch_type_2",
        "pitch_type",   # <-- IMPORTANT: this is the "action" we choose
        "outcome",      # <-- IMPORTANT: what happened after choosing it
    ]
    out_df = df[out_cols].copy()

    print("[OUT]", out_df.shape)
    print(out_df["outcome"].value_counts().head(12))

    out_df.to_parquet(args.out, index=False)
    print("[SAVE]", args.out)


if __name__ == "__main__":
    main()
