# src/prep.py
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd


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
]


def load_parquets(indir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(indir, "statcast_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in: {indir}")

    dfs = []
    for p in paths:
        df = pd.read_parquet(p, columns=KEEP_COLS)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    return out


def add_prev_pitch_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort so shifting works correctly inside each at-bat
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()

    group_keys = ["game_pk", "at_bat_number"]
    df["prev_pitch_type_1"] = df.groupby(group_keys)["pitch_type"].shift(1)
    df["prev_pitch_type_2"] = df.groupby(group_keys)["pitch_type"].shift(2)

    # For the first pitches of an AB, prev_pitch_type will be NaN; fill with a token
    df["prev_pitch_type_1"] = df["prev_pitch_type_1"].fillna("NONE")
    df["prev_pitch_type_2"] = df["prev_pitch_type_2"].fillna("NONE")

    return df


def make_next_pitch_label(df: pd.DataFrame) -> pd.DataFrame:
    # Label = next pitch type within the same at-bat
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).copy()
    group_keys = ["game_pk", "at_bat_number"]
    df["next_pitch_type"] = df.groupby(group_keys)["pitch_type"].shift(-1)

    # Drop last pitch of each at-bat (no next pitch)
    df = df.dropna(subset=["next_pitch_type"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare Statcast Parquet files for PitchSense modeling.")
    parser.add_argument("--indir", default="data/raw", help="Directory containing statcast_*.parquet files")
    parser.add_argument("--out", default="data/processed/pitchsense_v0.parquet", help="Output parquet path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = load_parquets(args.indir)
    print("[LOAD]", df.shape)

    # Basic cleanup
    df = df.dropna(subset=["pitch_type", "balls", "strikes", "stand", "p_throws"])
    print("[CLEAN]", df.shape)

    df = add_prev_pitch_features(df)
    df = make_next_pitch_label(df)

    # Keep only what we need for v0 modeling
    final_cols = [
        "stand",
        "p_throws",
        "balls",
        "strikes",
        "prev_pitch_type_1",
        "prev_pitch_type_2",
        "next_pitch_type",
    ]
    out_df = df[final_cols].copy()
    print("[FINAL]", out_df.shape)
    print(out_df.head(10))

    out_df.to_parquet(args.out, index=False)
    print("[SAVE]", args.out)


if __name__ == "__main__":
    main()
