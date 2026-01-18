"""
recommend.py

What this does:
1) Loads the labeled CSV created by one_button_pipeline.py (from ./data/).
2) Engineers features for pitch recommendation (count, prev pitch, etc.).
3) Trains a classifier to predict "effective" (1/0) for the *next* pitch.
4) Provides a CLI to recommend a greedy pitch sequence for this batter.

Run:
    source .venv/bin/activate
    python recommend.py

It will prompt for:
- Batter first/last name and season (to locate ./data/<last>_<first>_<season>_cleaned_labeled.csv)
- Number of pitches to recommend (default 3)
- Starting count (balls, strikes), default (0, 0)

Dependencies (install inside your venv if needed):
    pip install scikit-learn pandas numpy
"""

from pathlib import Path
import sys
import json
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Try XGBoost if available; otherwise fallback to GradientBoosting
try:
    from xgboost import XGBClassifier  # optional
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


@dataclass
class BatterKey:
    first: str
    last: str
    season: str

    @property
    def base(self) -> str:
        return f"{self.last.lower()}_{self.first.lower()}_{self.season}"

    @property
    def labeled_path(self) -> Path:
        return DATA_DIR / f"{self.base}_cleaned_labeled.csv"


def prompt_user() -> BatterKey:
    print("=== Recommend  Pitches===")
    first = input("Batter first name: ").strip()
    last = input("Batter last name: ").strip()
    season = input("Season year (e.g., 2024): ").strip() or "2024"
    return BatterKey(first=first, last=last, season=season)


def load_data(key: BatterKey) -> pd.DataFrame:
    path = key.labeled_path
    if not path.exists():
        print(f"ERROR: Labeled file not found: {path}")
        print("Run one_button_pipeline.py first for this batter & season.")
        sys.exit(1)
    df = pd.read_csv(path)
    if "effective" not in df.columns:
        print("ERROR: labeled CSV missing 'effective' column.")
        sys.exit(1)
    return df


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous-pitch features per at-bat, using game_date + inning + batter + pitcher
    and a running pitch index within each at-bat approximation.
    Statcast does not always include a unique at-bat id across all exports,
    so we'll approximate an at-bat key.
    """
    df = df.copy()

    # Best-effort at-bat key (can refine later)
    # If 'at_bat_number' exists, use it; otherwise approximate
    if 'at_bat_number' in df.columns:
        ab_key = ['game_date', 'batter', 'pitcher', 'inning', 'at_bat_number']
    else:
        ab_key = ['game_date', 'batter', 'pitcher', 'inning']

    df.sort_values(by=ab_key + ['balls', 'strikes'], inplace=True, kind='mergesort')

    # Previous pitch features
    for col in ['pitch_type', 'zone', 'release_speed', 'plate_x', 'plate_z', 'description']:
        prev_name = f'prev_{col}'
        df[prev_name] = df.groupby(ab_key)[col].shift(1)

    # Impute first-pitch prev_* values
    df['prev_pitch_type'] = df['prev_pitch_type'].fillna('NONE')
    df['prev_zone'] = df['prev_zone'].fillna(0).astype(float)
    df['prev_release_speed'] = df['prev_release_speed'].fillna(df['release_speed'].median())
    df['prev_plate_x'] = df['prev_plate_x'].fillna(0.0)
    df['prev_plate_z'] = df['prev_plate_z'].fillna(0.0)
    df['prev_description'] = df['prev_description'].fillna('NONE')

    # Coerce types
    for c in ['zone', 'prev_zone', 'balls', 'strikes', 'inning']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    # Compact descriptions (optional)
    df['desc_simple'] = df['description'].fillna('none').replace({
        'foul_tip': 'foul',
        'foul_bunt': 'foul',
        'swinging_strike_blocked': 'swinging_strike',
    })

    return df


def build_pipeline(categorical_features: List[str], numeric_features: List[str]) -> Pipeline:
    cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    pre = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_features),
            ('num', 'passthrough', numeric_features),
        ]
    )
    if HAS_XGB:
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric='logloss',
        )
    else:
        model = GradientBoostingClassifier(random_state=42)

    pipe = Pipeline(steps=[('pre', pre), ('clf', model)])
    return pipe


def train_model(df: pd.DataFrame, key: BatterKey) -> Tuple[Pipeline, dict]:
    # Features & label
    df = add_sequence_features(df)

    # Minimal drop of rows missing label
    df = df.dropna(subset=['effective'])

    categorical = [
        'pitch_type', 'pitch_name', 'stand', 'p_throws',
        'prev_pitch_type', 'prev_description'
    ]
    numeric = [
        'release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'zone',
        'balls', 'strikes', 'inning',
        'prev_release_speed', 'prev_plate_x', 'prev_plate_z', 'prev_zone'
    ]

    use_cols = categorical + numeric + ['effective']
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        print("NOTE: Some expected features are missing and will be skipped:", missing)
    present_cats = [c for c in categorical if c in df.columns]
    present_nums = [c for c in numeric if c in df.columns]

    X = df[present_cats + present_nums]
    y = df['effective'].astype(int)

    # Train/test split by shuffle (quick baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(present_cats, present_nums)
    pipe.fit(X_train, y_train)

    # Eval
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = float('nan')
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        'auc': auc,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'support': int(report['1']['support']),
    }
    print("\n=== Validation Metrics (class=effective=1) ===")
    print(json.dumps(metrics, indent=2))

    # Save
    model_path = MODELS_DIR / f"{key.base}_pitch_effectiveness_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'pipeline': pipe, 'categorical': present_cats, 'numeric': present_nums}, f)
    print(f"\nSaved model to: {model_path}")
    return pipe, metrics


def recommend_next(pipe: Pipeline, template_row: pd.Series,
                   candidates: List[Tuple[str, int]]) -> Tuple[Tuple[str, int], float]:
    """
    Given a model and a base row (context), score candidate (pitch_type, zone) pairs.
    Returns the best candidate and its predicted probability of effectiveness.
    """
    best = None
    best_p = -1.0
    for pitch_type, zone in candidates:
        row = template_row.copy()
        row['pitch_type'] = pitch_type
        row['zone'] = zone
        # carry previous pitch context as-is (already in template_row)
        proba = pipe.predict_proba(pd.DataFrame([row]))[0, 1]
        if proba > best_p:
            best_p = proba
            best = (pitch_type, zone)
    return best, best_p


def greedy_sequence(pipe: Pipeline, df_context: pd.DataFrame,
                    length: int = 3,
                    start_balls: int = 0, start_strikes: int = 0) -> List[dict]:
    """
    Build a greedy sequence of pitches by selecting the best next (pitch_type, zone)
    at each step for the given batter context.
    We derive candidate pitches and zones from the batter's historical data.
    """
    seq = []

    # Template: mean/median of numeric context + common categorical defaults
    template = {}
    for col in ['release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z']:
        if col in df_context.columns:
            template[col] = float(df_context[col].median())
        else:
            template[col] = 0.0
    template['pitch_name'] = 'Unknown'
    template['stand'] = df_context['stand'].mode().iat[0] if 'stand' in df_context.columns and not df_context['stand'].empty else 'R'
    template['p_throws'] = df_context['p_throws'].mode().iat[0] if 'p_throws' in df_context.columns and not df_context['p_throws'].empty else 'R'
    template['balls'] = int(start_balls)
    template['strikes'] = int(start_strikes)
    template['inning'] = int(df_context['inning'].median()) if 'inning' in df_context.columns else 1
    template['prev_pitch_type'] = 'NONE'
    template['prev_description'] = 'NONE'
    template['prev_release_speed'] = template['release_speed']
    template['prev_plate_x'] = template['plate_x']
    template['prev_plate_z'] = template['plate_z']
    template['prev_zone'] = 0

    # Candidate sets from historical distribution (top 5 pitch types, zones 1..14 seen)
    pitch_types = df_context['pitch_type'].dropna().value_counts().head(5).index.tolist() or ['FF', 'SL', 'CH', 'CU']
    zones_hist = sorted(df_context['zone'].dropna().astype(int).unique().tolist())
    zones = [z for z in zones_hist if 1 <= z <= 14] or list(range(1, 15))
    candidates = [(pt, z) for pt in pitch_types for z in zones]

    for i in range(length):
        best, p = recommend_next(pipe, pd.Series(template), candidates)
        if best is None:
            break
        pitch_type, zone = best

        # Append step
        seq.append({
            'step': i + 1,
            'pitch_type': pitch_type,
            'zone': int(zone),
            'predicted_effective_prob': float(round(p, 4)),
            'count_before': f"{template['balls']}-{template['strikes']}",
        })

        # Update prev_* and count for next step (simple heuristic):
        template['prev_pitch_type'] = pitch_type
        template['prev_zone'] = int(zone)
        template['prev_release_speed'] = template['release_speed']
        template['prev_plate_x'] = template['plate_x']
        template['prev_plate_z'] = template['plate_z']
        template['prev_description'] = 'called_strike'  # optimistic carry

        # Simple count update: treat high prob as strike, otherwise ball
        if p >= 0.5:
            template['strikes'] = min(2, template['strikes'] + 1)
        else:
            template['balls'] = min(3, template['balls'] + 1)

        # Early stop if 2 strikes reached (next pitch would be K if effective)
        if template['strikes'] == 2 and i < length - 1:
            # leave last recommendation to try to "put away" the batter
            pass

    return seq


def cli():
    key = prompt_user()
    df = load_data(key)
    print(f"Loaded {len(df)} labeled pitches for {key.first} {key.last} ({key.season}).")

    pipe, _metrics = train_model(df, key)

    # Context subset for candidate generation (use the same batter-season frame)
    df_context = df.copy()

    try:
        length = int(input("How many pitches to recommend? [3]: ").strip() or "3")
    except Exception:
        length = 3
    try:
        cnt = input("Starting count (balls-strikes) [0-0]: ").strip() or "0-0"
        b, s = cnt.split("-")
        start_balls, start_strikes = int(b), int(s)
    except Exception:
        start_balls, start_strikes = 0, 0

    seq = greedy_sequence(pipe, df_context, length=length,
                          start_balls=start_balls, start_strikes=start_strikes)

    print("\n=== Recommended Pitch Sequence  ===")
    for step in seq:
        print(f"Step {step['step']}: {step['pitch_type']} to zone {step['zone']} "
              f"(p_effective={step['predicted_effective_prob']}) at count {step['count_before']}")

    # Save sequence and metadata
    out = {
        'batter': {'first': key.first, 'last': key.last, 'season': key.season},
        'sequence': seq,
    }
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    out_path = model_dir / f"{key.base}_recommended_sequence.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved sequence JSON to: {out_path}")


if __name__ == "__main__":
    cli()
