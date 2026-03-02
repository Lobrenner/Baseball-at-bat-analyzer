PitchSense 

AI-Assisted Pitch Sequencing Using Statcast Data

Overview

- PitchSense is a PyTorch-based system that analyzes MLB Statcast pitch-by-pitch data to recommend optimal pitch selections and multi-pitch sequences for pitchers. The goal is to support pitcher decision-making by modeling how pitch type, count, handedness, and pitch history influence outcomes such as strikeouts, balls, and balls in play.
  This project focuses on sequence-aware decision making, rather than treating each pitch independently.

Key Features
- Statcast-based data pipeline using pybaseball
- Neural network outcome model trained in PyTorch
- Pitch sequence planning via beam search
- Optimization toward pitcher-favorable outcomes (e.g. strikeouts)
- Repeat penalties and pitch tunneling heuristics
- Pitcher-specific arsenals (only recommends pitches a pitcher actually throws)

Project Structure
PitchSense/
├── src/
│   ├── download.py              # Download raw Statcast data
│   ├── prep_outcomes.py                  # Clean & preprocess data
│   ├── train_outcomes.py        # Train PyTorch outcome model
│   ├── recommend_sequence.py    # Recommend multi-pitch sequences
│   └── train.py                 # Experimental / legacy training script
├── data/
│   ├── raw/                     # Raw downloaded Statcast data (ignored by git)
│   └── processed/               # Cleaned model-ready data (ignored by git)
├── models/                      # Trained PyTorch models (ignored by git)
└── README.md

Installation
Requirements

Python 3.10+

pybaseball

pandas

numpy

torch

pyarrow

Pipeline Usage
1. Download Statcast Data (download.py)

Downloads pitch-by-pitch data for a date range.

python src/download.py --start 2024-06-01 --end 2024-06-07

2. Preprocess Data (prep_outcomes.py)

Cleans raw data, normalizes outcomes, and prepares features for training.

python src/prep_outcomes.py --indir data/raw --out data/processed/outcomes_with_outs.parquet


3. Train Outcome Model (train_outcomes.py)

Trains a PyTorch classification model that predicts pitch outcomes given context.

python src/train_outcomes.py --data data/processed/outcomes_with_outs.parquet --outdir models/Test_v9_outs

Outputs:

models/pitchsense_outcomes_pitcher_v1/
 ├── model.pt
 └── encoders.json


4. Recommend Pitch Sequences (recommend_sequence.py)

Uses beam search to recommend a sequence of pitches optimized for pitcher success.

python src/recommend_sequence.py `
  --modeldir models/Test_v9_outs `
  --data data/processed/outcomes_with_outs.parquet `
  --pitcher 669373 `
  --batter 669257 `
  --stand R `
  --p_throws L `
  --balls 1 `
  --strikes 2 `
  --outs 2 `
  --inning 4 `
  --tto 3 `
  --on_1b `
  --beam_width 8 `
  --depth 4 `
  --json


Modeling Approach

- Inputs: pitcher ID, batter handedness, pitcher handedness, count, previous pitches

- Outputs: probabilities for pitch outcomes (K, ball, hit, etc.)

- Sequence planning: beam search with:

- repeat pitch penalties

- tunneling bonuses

- pitcher-specific pitch constraints

- The system currently models pitcher vs generic batter, with extensions planned.

Planned Extensions

- Pitch location modeling (zone buckets) #complete

- Batter-specific matchups #complete

- Maybe add three general classifications of a batter (Aggressive, Mix, passive) so a sequence is generated for a specific archetype

- Larger training data set #complete kinda, could still be bigger

- Implement "stuff-awareness" (some pitchers may have better pitches than others even if they throw the same pitch type). Model should understand understand when a pitcher has a specific unique/dominant pitch that would be more effective than if it came form a different pitcher

Notes

...
