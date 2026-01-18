from pathlib import Path
from datetime import date
import sys

try:
    from pybaseball import playerid_lookup, statcast_batter
except Exception as e:
    print("ERROR: pybaseball is required. Activate your venv and install with:")
    print("       pip install pybaseball")
    raise

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

def prompt_user():
    print("=== At-Bat Analyzer ===")
    first = input("Enter batter's first name: ").strip()
    last = input("Enter batter's last name: ").strip()
    season = input(f"Enter season year (e.g., {date.today().year - 1}): ").strip()
    if not season.isdigit() or len(season) != 4:
        print("Invalid year; defaulting to last season.")
        season = str(date.today().year - 1)
    return first, last, season

def fetch_statcast(first, last, season):
    print(f"\n[1/5] Looking up MLBAM ID for {first} {last}...")
    info = playerid_lookup(last, first)
    if info.empty:
        print(f"ERROR: No player found for {first} {last}. Check spelling.")
        sys.exit(1)
    batter_id = int(info['key_mlbam'].values[0])
    print(f"    -> MLBAM ID: {batter_id}")

    start = f"{season}-03-01"
    end   = f"{season}-11-30"
    print(f"[2/5] Fetching Statcast pitches for {season} ({start} to {end})...")
    df = statcast_batter(start, end, batter_id)
    print(f"    -> Retrieved {len(df)} rows")
    if len(df) == 0:
        print("WARNING: No data returned for this season/player.")
    return df, batter_id

def clean_columns(df):
    print("[3/5] Cleaning to core columns...")
    keep = [
        'game_date', 'pitch_type', 'pitch_name', 'release_speed',
        'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone',
        'description', 'events', 'balls', 'strikes', 'outs_when_up',
        'inning', 'stand', 'p_throws', 'launch_speed', 'launch_angle',
        'hit_distance_sc', 'home_team', 'away_team', 'batter', 'pitcher'
    ]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        print(f"    -> NOTE: Missing columns (will skip): {missing}")
    cleaned = df[[c for c in keep if c in df.columns]].copy()
    return cleaned

def label_effective(df):
    print("[4/5] Labeling pitch effectiveness...")
    effective_set = {'swinging_strike', 'swinging_strike_blocked', 'called_strike'}
    df['effective'] = df['description'].isin(effective_set).astype(int)
    if 'strikes' in df.columns:
        mask = (df['description'] == 'foul') & (df['strikes'] == 2)
        df.loc[mask, 'effective'] = 1
    return df

def save_outputs(df_clean, first, last, season):
    base = f"{last.lower()}_{first.lower()}_{season}"
    cleaned_path = DATA_DIR / f"{base}_cleaned.csv"
    labeled_path = DATA_DIR / f"{base}_cleaned_labeled.csv"

    df_clean.to_csv(cleaned_path, index=False)
    print(f"    -> Saved cleaned CSV: {cleaned_path}")

    df_labeled = label_effective(df_clean.copy())
    df_labeled.to_csv(labeled_path, index=False)
    print(f"    -> Saved labeled CSV: {labeled_path}")

    # Save path for downstream tools
    with open(BASE_DIR / "last_batter_filename.txt", "w") as f:
        f.write(str(cleaned_path))

    return cleaned_path, labeled_path

def summary(df):
    print("\n[5/5] Quick summary")
    total = len(df)
    eff = int(df.get('effective', pd.Series([0]*total)).sum())
    print(f"    Total pitches: {total}")
    print(f"    Effective pitches: {eff}")
    if 'pitch_type' in df.columns:
        print("    Top pitch types:")
        print(df['pitch_type'].value_counts().head(5).to_string())

def main():
    first, last, season = prompt_user()
    raw, batter_id = fetch_statcast(first, last, season)
    cleaned = clean_columns(raw)
    cleaned_path, labeled_path = save_outputs(cleaned, first, last, season)

    df_labeled = pd.read_csv(labeled_path)
    summary(df_labeled)

    print("\nDone! Files created:")
    print(f"  - {cleaned_path}")
    print(f"  - {labeled_path}")

if __name__ == '__main__':
    main()
