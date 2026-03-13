"""
lanl_redteam_parser.py
Reconstructs redteam ground truth from broken extraction.
The redteam.txt folder has each data ROW as a filename (extraction bug).
We read those filenames back as the actual data.
Format: time,user@domain,src_computer,dst_computer
"""

import os
import pandas as pd

REDTEAM_FOLDER = r"C:\Users\rosha\Documents\PIRS\lanl_data\redteam.txt"
OUTPUT_PATH = r"C:\Users\rosha\Documents\PIRS\lanl_data\redteam_parsed.csv"


def parse_redteam():
    rows = []
    entries = os.listdir(REDTEAM_FOLDER)
    print(f"Found {len(entries)} redteam entries (filenames)")

    for name in entries:
        name = name.strip()
        if not name:
            continue
        parts = name.split(",")
        if len(parts) != 4:
            print(f"  Skipping malformed: {name}")
            continue
        try:
            time_sec = int(parts[0])
            user = parts[1].strip()
            src_computer = parts[2].strip()
            dst_computer = parts[3].strip()
            day = time_sec // 86400
            hour = (time_sec % 86400) // 3600
            # Extract clean user ID (e.g. "U66" from "U66@DOM1")
            user_clean = user.split("@")[0]
            rows.append({
                "time": time_sec,
                "day": day,
                "hour": hour,
                "user_raw": user,
                "user": user_clean,
                "src_computer": src_computer,
                "dst_computer": dst_computer,
            })
        except Exception as e:
            print(f"  Error parsing '{name}': {e}")

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)

    print(f"\nParsed {len(df)} redteam events")
    print(f"Unique red team users: {df['user'].nunique()}")
    print(df['user'].value_counts().head(20))
    print(f"\nDay range: {df['day'].min()} - {df['day'].max()}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    df = parse_redteam()
    print("\nSample:")
    print(df.head(10).to_string())
