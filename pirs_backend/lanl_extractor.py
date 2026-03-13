"""
lanl_extractor.py
Reads auth.txt (69GB) in chunks and extracts per-user-per-day behavioral features.
auth.txt columns: time, src_user, dst_user, src_computer, dst_computer,
                  auth_type, logon_type, auth_orientation, success
"""

import pandas as pd
import numpy as np
import os
import time

AUTH_PATH = r"C:\Users\rosha\Documents\PIRS\lanl_data\auth.txt\auth.txt"
OUTPUT_PATH = r"C:\Users\rosha\Documents\PIRS\lanl_data\lanl_features.csv"
CHUNK_SIZE = 500_000
WORK_HOUR_START = 8   # 8am
WORK_HOUR_END = 17    # 5pm
SECONDS_PER_DAY = 86_400

AUTH_COLS = [
    "time", "src_user", "dst_user", "src_computer", "dst_computer",
    "auth_type", "logon_type", "auth_orientation", "success"
]


def extract_user(user_str):
    """Extract clean user ID: 'U66@DOM1' -> 'U66'"""
    if pd.isna(user_str):
        return None
    return str(user_str).split("@")[0]


def process_chunk(chunk):
    """Aggregate a chunk into per-user-per-day feature rows."""
    chunk.columns = AUTH_COLS

    # Convert time to day and hour
    chunk["time"] = pd.to_numeric(chunk["time"], errors="coerce")
    chunk = chunk.dropna(subset=["time"])
    chunk["day"] = (chunk["time"] // SECONDS_PER_DAY).astype(int)
    chunk["hour"] = ((chunk["time"] % SECONDS_PER_DAY) // 3600).astype(int)

    # Clean user ID from src_user (the acting user)
    chunk["user"] = chunk["src_user"].apply(extract_user)
    chunk = chunk[chunk["user"].notna()]

    # Drop machine accounts (end with $) and ANONYMOUS
    chunk = chunk[~chunk["user"].str.endswith("$")]
    chunk = chunk[~chunk["user"].str.upper().str.contains("ANONYMOUS")]
    chunk = chunk[~chunk["user"].str.upper().str.contains("SYSTEM")]

    # Binary flags
    chunk["is_logon"] = (chunk["auth_orientation"] == "LogOn").astype(int)
    chunk["is_logoff"] = (chunk["auth_orientation"] == "LogOff").astype(int)
    chunk["is_failed"] = (chunk["success"] == "Fail").astype(int)
    chunk["is_workhour"] = (
        (chunk["hour"] >= WORK_HOUR_START) & (chunk["hour"] < WORK_HOUR_END)
    ).astype(int)
    chunk["is_afterhour"] = (1 - chunk["is_workhour"]).astype(int)
    chunk["is_ntlm"] = (chunk["auth_type"] == "NTLM").astype(int)
    chunk["is_kerberos"] = (chunk["auth_type"] == "Kerberos").astype(int)
    chunk["is_network"] = (chunk["logon_type"] == "Network").astype(int)
    chunk["is_service"] = (chunk["logon_type"] == "Service").astype(int)
    chunk["is_interactive"] = (chunk["logon_type"] == "Interactive").astype(int)
    chunk["is_lateral"] = (
        chunk["src_computer"] != chunk["dst_computer"]
    ).astype(int)

    grp = chunk.groupby(["user", "day"])

    agg = grp.agg(
        n_auth=("time", "count"),
        n_logon=("is_logon", "sum"),
        n_logoff=("is_logoff", "sum"),
        n_failed=("is_failed", "sum"),
        n_unique_dst=("dst_computer", "nunique"),
        n_unique_src=("src_computer", "nunique"),
        n_workhour=("is_workhour", "sum"),
        n_afterhour=("is_afterhour", "sum"),
        n_ntlm=("is_ntlm", "sum"),
        n_kerberos=("is_kerberos", "sum"),
        n_network=("is_network", "sum"),
        n_service=("is_service", "sum"),
        n_interactive=("is_interactive", "sum"),
        n_lateral=("is_lateral", "sum"),
    ).reset_index()

    return agg


def run_extraction():
    print(f"Starting LANL auth.txt extraction")
    print(f"File: {AUTH_PATH}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    accumulated = []
    chunk_num = 0
    total_rows = 0
    t0 = time.time()

    reader = pd.read_csv(
        AUTH_PATH,
        header=None,
        chunksize=CHUNK_SIZE,
        dtype=str,
        on_bad_lines="skip",
    )

    for chunk in reader:
        chunk_num += 1
        total_rows += len(chunk)

        agg = process_chunk(chunk)
        accumulated.append(agg)

        elapsed = time.time() - t0
        rate = total_rows / elapsed / 1_000_000
        print(
            f"  Chunk {chunk_num:3d} | Rows: {total_rows:12,} | "
            f"Speed: {rate:.2f}M rows/s | Elapsed: {elapsed:.0f}s"
        )

        # Merge accumulated every 20 chunks to keep memory in check
        if len(accumulated) >= 20:
            merged = pd.concat(accumulated, ignore_index=True)
            accumulated = [
                merged.groupby(["user", "day"]).sum().reset_index()
            ]

    print(f"\nFinalizing aggregation...")
    if accumulated:
        final = pd.concat(accumulated, ignore_index=True)
        final = final.groupby(["user", "day"]).sum().reset_index()
    else:
        print("No data extracted!")
        return

    # Sort
    final = final.sort_values(["user", "day"]).reset_index(drop=True)

    # Add derived ratio features
    final["fail_rate"] = final["n_failed"] / (final["n_auth"] + 1)
    final["afterhour_rate"] = final["n_afterhour"] / (final["n_auth"] + 1)
    final["lateral_rate"] = final["n_lateral"] / (final["n_auth"] + 1)
    final["ntlm_rate"] = final["n_ntlm"] / (final["n_auth"] + 1)
    final["dst_diversity"] = final["n_unique_dst"] / (final["n_auth"] + 1)

    total_time = time.time() - t0
    print(f"\nExtraction complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total rows processed: {total_rows:,}")
    print(f"Unique users: {final['user'].nunique():,}")
    print(f"Unique days: {final['day'].nunique()}")
    print(f"Total user-day records: {len(final):,}")

    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    return final


if __name__ == "__main__":
    df = run_extraction()
    if df is not None:
        print("\nFeature sample:")
        print(df.head(10).to_string())
        print("\nFeature stats:")
        print(df.describe().to_string())
