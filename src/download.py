#Download statcast data pitch by pitch data for a data range
# src/download.py

from __future__ import annotations
from pybaseball import statcast
import pandas as pd

import argparse
import os
from datetime import datetime, timedelta
import time

def daterange(start_date: str, end_date: str) :
    '''YYYY-MM-DD get start and end dates'''
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end_date must be later than sstart_date")
    
    d = start
    while d <= end:
        yield d.isoformat()
        d += timedelta(days=1)

def download_day(day: str) -> pd.DataFrame:
    '''Download all statcast pitch data for a day'''
    return statcast (day, day)

def main():
    parser = argparse.ArgumentParser(description="Download Statcast data day-by-day and cache as parquet")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--outdir", default="data/raw", help="Output directory for Parquet files")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between days")
    parser.add_argument("--retries", type=int, default=3, help="Retries per day if download fails")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    for day in daterange(args.start, args.end):
        outpath = os.path.join(args.outdir, f"statcast_{day}.parquet")

        if os.path.exists(outpath) and not args.force:
            print(f"[SKIP] {day} already exists: {outpath}")
            continue

        ok = False
        for attempt in range (1, args.retries + 1):
            try:
                print(f"[DL] {day} (attempt {attempt}/{args.retries})")
                df = download_day(day)

                if df is None or len(df) == 0:
                    print(f"[WARN] {day} returned 0 rows. Saving anyway.")
                else:
                    print(f"[OK] {day} rows={len(df)} cols={len(df.columns)}")

                df.to_parquet(outpath, index=False)
                print(f"[SAVE] {outpath}")
                ok = True
                break
            except Exception as e:
                print(f"[ERR] {day} attempt {attempt} failed: {e}")
                if attempt < args.retries:
                    time.sleep(5)
                

        if not ok:
            print(f"[FAIL] Giving up on {day}. Moving on.")

    time.sleep(args.sleep)

if __name__ == "__main__":
    main()