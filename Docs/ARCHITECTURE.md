# PitchSense – Architecture & API Notes

This document explains the internal structure of PitchSense

## System Overview

PitchSense is divided into two layers:

### Offline training
- `download.py` – fetches raw Statcast data
- `prep_outcomes.py` – cleans data and defines outcome labels
- `train_outcomes.py` – trains the PyTorch outcome model

### Online inference
Should be called by API
- `recommend_next.py`
- `recommend_sequence.py`

Note* recommend_next is not run idividually, but it is used inside of recommend_sequence

The API should ONLY interact with the online layer.

---

## Data Flow

Game state
(pitcher, handedness, count, pitch history)
→ PyTorch model inference
→ scoring + sequencing logic
→ ranked pitch recommendations

---

## Entry Points

### recommend_next.py
Returns a ranked list of candidate pitches for the *next pitch only*.

**Inputs**
- pitcher (MLBAM id, int)
- stand (L / R)
- p_throws (L / R)
- balls (0–3)
- strikes (0–2)
- prev1 (optional pitch type)
- prev2 (optional pitch type)

**Output**
- ordered pitches with outcome probabilities and score

---

### recommend_sequence.py
Returns ranked pitch sequences using beam search.

**Inputs**
- Same as recommend_next.py
- beam_width (int)
- depth (int)

**Output**
- ordered sequences
- cumulative score per sequence
- optional per-step explanation fields

Scores are relative and only meaningful within a single request.

---

## Model Notes

- Model is a PyTorch classifier trained to predict pitch outcomes
- Inference is CPU-only by default
- Model weights and encoders are loaded from disk
- No training occurs during inference

---


