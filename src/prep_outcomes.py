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
    "balls", #count
    "strikes", #count
    "pitch_type",
    "description",
    "events", #what happened after that pitch
    "zone", # pitch location
    "inning",
    "on_1b",
    "on_2b",
    "on_3b",
    "n_thruorder_pitcher",
]

OPTIONAL_COLS = ["outs_when_up"] #optional because sometimes not present in data and don't want to skip whole files

# Outcome mapping helpers
HIT_EVENTS = {"single", "double", "triple", "home_run"}
WALK_EVENTS = {"walk", "intent_walk", "hit_by_pitch"}
# Baseball playes that result in outs
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
    skipped = 0

    for p in paths:
        try:
            df0 = pd.read_parquet(p)  # read whole file to validate schema
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)} unreadable parquet: {e}")
            skipped += 1
            continue

        if df0 is None or len(df0) == 0:
            print(f"[SKIP] {os.path.basename(p)} is empty")
            skipped += 1
            continue

        missing = [c for c in KEEP_COLS if c not in df0.columns]
        if missing:
            print(f"[SKIP] {os.path.basename(p)} missing required cols: {missing}")
            skipped += 1
            continue

        cols = KEEP_COLS + [c for c in OPTIONAL_COLS if c in df0.columns]
        dfs.append(df0[cols])

    if not dfs:
        raise RuntimeError(f"All parquet files were empty/bad in: {indir}")

    out = pd.concat(dfs, ignore_index=True)
    print(f"[LOAD] kept={len(dfs)} skipped={skipped} rows={len(out)}")
    return out



def add_prev_action_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    keys = ["game_pk", "at_bat_number"]

    #creates previous action features within the same at-bat
    df["prev_action_1"] = df.groupby(keys)["pitch_action"].shift(1).fillna("NONE")
    df["prev_action_2"] = df.groupby(keys)["pitch_action"].shift(2).fillna("NONE")
    return df


def label_outcome(row: pd.Series) -> str:
    """
    Returns a label representing the outcome AFTER this pitch.
    Terminal outcomes are preferred when `events` is present.
    Otherwise we fall back to pitch-by-pitch `description`.
    """
    ev = row.get("events")
    desc = row.get("description")

    #  Terminal outcomes from events (end of PA outcomes)
    if isinstance(ev, str) and ev:
        if ev == "strikeout" or ev == "strikeout_double_play":
            return "K"
        if ev in WALK_EVENTS:
            return "BB"
        if ev in HIT_EVENTS:
            return "HIT"
        if ev in BIP_OUT_EVENTS:
            return "BIP_OUT"
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

    def map_inning_bucket(inning: float | int) -> str:
        try:
            inn = int(inning)
        except Exception:
            return "early"  # default to early if there is something weird
        if inn <= 3:
            return "early"
        if inn <= 6:
            return "mid"
        if inn <= 9:
            return "late"
        return "extras"

    df["inning_bucket"] = df["inning"].apply(map_inning_bucket)

    def map_tto_bucket(n) -> str:
    # 1=1st time thru order, 2=2nd, 3=3rd, ... never has 0th time thru order
        try:
            val = int(n)
        except Exception:
            return "1"  # safe default

        if val <= 1:
            return "1"
        if val == 2:
            return "2"
        return "3+"

    df["tto_bucket"] = df["n_thruorder_pitcher"].apply(map_tto_bucket)

    df["on_1b_bool"] = df["on_1b"].notna()
    df["on_2b_bool"] = df["on_2b"].notna()
    df["on_3b_bool"] = df["on_3b"].notna()

    df["base_state"] = (
        df["on_1b_bool"].map({True: "1", False: "-"})
        + df["on_2b_bool"].map({True: "2", False: "-"})
        + df["on_3b_bool"].map({True: "3", False: "-"})
    )

    df["runners_count"] = df[["on_1b_bool","on_2b_bool","on_3b_bool"]].sum(axis=1).astype(int)

    #add number labeled pitch zone
    df["zone"] = pd.to_numeric(df["zone"], errors = "coerce")

    #location, this takes the zone stat from raw dataset and makes location. 
    #strikezone includes 1-9 so we are omitting 11-14 (10 doesn't exist ig) because they are balls and could be bad pitches, probably should include them later though when the model is better for pitch tunneling
    df["loc_bucket"] = df["zone"].apply(
        lambda z: f"Z{int(z)}" if pd.notna(z) and 1 <= z <= 9 else "OZ" #OZ = outzone aka ball or unrecorded
    )

    df["pitch_action"] = df["pitch_type"] + "|" + df["loc_bucket"] #makes a colum that describes a pitch with its type and location ex. FF|Z1 this would be a fastball top left of the zone

    print(df["loc_bucket"].value_counts(dropna=False).head(15))

    df = add_prev_action_features(df)

    # apply outcome labeling per pitch
    df["outcome"] = df.apply(label_outcome, axis=1)
    df["outcome"] = df["outcome"].replace({"OTHER_PITCH": "OTHER", "OTHER_EVENT": "OTHER"})

    #turn outs_when_up into just outs
    if "outs_when_up" in df.columns:
        df["outs"] = pd.to_numeric(df["outs_when_up"], errors="coerce").fillna(0).astype(int)
    else:
        print("[WARN] outs_when_up not found; setting outs=0 for all rows (model won't learn outs).")
        df["outs"] = 0

    df["outs"] = df["outs"].clip(0, 2)
    # Keep a modeling-friendly subset
    out_cols = [
        "pitcher",
        "batter",
        "stand",
        "p_throws",
        "balls",
        "strikes",
        "prev_action_1",   #previous pitch
        "prev_action_2",   #two pitches prior
        "pitch_action",   #this is the "action" we choose
        "outcome",      # what was the result of pitch being thrown
        "inning_bucket",
        "base_state",
        "runners_count",
        "tto_bucket",  
        "outs"
    ]
    out_df = df[out_cols].copy()

    print("[OUT]", out_df.shape)
    print(out_df["outcome"].value_counts().head(12))

    out_df.to_parquet(args.out, index=False)
    print("[SAVE]", args.out)


if __name__ == "__main__":
    main()
