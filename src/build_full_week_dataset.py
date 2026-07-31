"""
build_full_week_dataset.py
---------------------------
Combines all 7 remapped/synthetic CICIDS2017-style day files (Mon-Sun) into
ONE full-scale vendor risk dataset (~3.1M rows), and recomputes the rolling
behavioral features (login_frequency_24h, failed_login_count,
avg_session_duration) as TRUE trailing 24-hour windows per vendor.
"""

import pandas as pd
import numpy as np
import random

INPUT_DIR = "Dataset"
FILES = [
    "monday_full_remapped_v2.csv",
    "tuesday_full_remapped.csv",
    "wednesday_full_remapped.csv",
    "thursday_full_remapped.csv",
    "friday_full_remapped.csv",
    "saturday_full_synthetic.csv",
    "sunday_full_synthetic.csv",
]

LOCAL_TZ = "Asia/Kolkata"  # change this if your data isn't IST

CANDIDATE_FORMATS = [
    "ISO8601",                     # weekday files: 2026-07-06T00:00:01+05:30
    "%d-%m-%Y %H:%M:%S.%f",        # weekend, with seconds + fraction
    "%d-%m-%Y %H:%M:%S",           # weekend, with seconds
    "%d-%m-%Y %H:%M",              # weekend, no seconds
]

def parse_timestamps(raw_series):
    remaining = raw_series.copy()
    result = pd.Series(pd.NaT, index=raw_series.index, dtype="object")

    for fmt in CANDIDATE_FORMATS:
        if remaining.isna().all() or (remaining == "").all():
            break
        mask_to_try = remaining.notna() & (remaining != "")
        if not mask_to_try.any():
            continue
        parsed_attempt = pd.to_datetime(remaining[mask_to_try], format=fmt, errors="coerce")
        got = parsed_attempt.notna()
        idx_hit = parsed_attempt[got].index
        result.loc[idx_hit] = parsed_attempt[got]
        remaining.loc[idx_hit] = pd.NA  

    # Fallback for anything no explicit format matched
    still_unparsed = remaining.notna() & (remaining != "")
    if still_unparsed.any():
        fallback = pd.to_datetime(remaining[still_unparsed], dayfirst=True, errors="coerce")
        result.loc[fallback.index] = fallback

    return pd.to_datetime(result, errors="coerce")


print("Loading all 7 days...")
frames = []
total_dropped = 0

for f in FILES:
    df_i = pd.read_csv(f"{INPUT_DIR}/{f}", low_memory=False)

    # --- FIX TRUNCATED TIMESTAMPS (Excel dropped date/hour) ---
    ts_str = df_i["timestamp"].astype(str).str.strip()
    
    # Truncated timestamps lack a date (no hyphen) but contain time parts (colon)
    # Examples: "49:05.0", "14:30"
    short_time_mask = ~ts_str.str.contains("-") & ts_str.str.contains(":") & (ts_str != "nan")
    
    if short_time_mask.any():
        # Get valid rows to determine what date this file represents
        valid_rows = ts_str[~short_time_mask & (ts_str != "nan")]
        if not valid_rows.empty:
            # Extract date prefix (everything before the first space or 'T')
            date_parts = valid_rows.str.extract(r'^([^ T]+)', expand=False)
            most_common_date = date_parts.mode()[0]  # e.g., "11-07-2026"

            # If it only has one colon (MM:SS), the "00:" hour was truncated. Restore it.
            one_colon_mask = short_time_mask & (ts_str.str.count(":") == 1)
            ts_str.loc[one_colon_mask] = "00:" + ts_str.loc[one_colon_mask]

            # Reattach the date to all truncated times
            ts_str.loc[short_time_mask] = most_common_date + " " + ts_str.loc[short_time_mask]

            # Apply fix back to dataframe before parsing
            df_i["timestamp"] = ts_str
    # -----------------------------------------------------------

    ts = parse_timestamps(df_i["timestamp"])

    bad_mask = ts.isna() & df_i["timestamp"].notna()
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        print(f"  !! {f}: {n_bad} still unparseable after all known formats, examples:")
        print("     ", df_i.loc[bad_mask, "timestamp"].head(10).to_list())
        df_i = df_i.loc[~bad_mask].copy()
        ts = ts.loc[~bad_mask]
        total_dropped += n_bad

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(LOCAL_TZ)
    else:
        ts = ts.dt.tz_convert(LOCAL_TZ)

    df_i["timestamp_dt"] = ts
    frames.append(df_i)
    print(f"  {f}: {len(df_i):,} rows kept")

df = pd.concat(frames, ignore_index=True)
print(f"\nTotal combined rows: {len(df):,}  (dropped {total_dropped} genuinely unparseable rows)")

# ---- sort globally on the now-consistent tz-aware datetime ----
df = df.sort_values(["vendor_id", "timestamp_dt"]).reset_index(drop=True)

# ---- off_hours_flag ----
np.random.seed(99)
random.seed(99)
VENDOR_POOL = sorted(df["vendor_id"].unique())
NORMAL_HOURS = {v: (random.randint(6, 9), random.randint(16, 19)) for v in VENDOR_POOL}

df["off_hours_flag"] = df.apply(
    lambda r: 0 if NORMAL_HOURS[r["vendor_id"]][0] <= r["timestamp_dt"].hour < NORMAL_HOURS[r["vendor_id"]][1] else 1,
    axis=1
)

# ---- TRUE trailing 24h rolling features ----
print("Computing rolling 24h behavioral features per vendor (this is the slow step)...")

df["is_login"] = (df["action_type"] == "login").astype(int)
df["is_failed_login"] = ((df["action_type"] == "login") & (df["auth_result"] == "failure")).astype(int)
df["session_seconds"] = df["flow_duration"] / 1_000_000

df = df.set_index("timestamp_dt")

def rolling_24h(group):
    vendor_name = group.name
    group = group.sort_index()
    group["login_frequency_24h"] = group["is_login"].rolling("24h").sum().astype(int)
    group["failed_login_count"] = group["is_failed_login"].rolling("24h").sum().astype(int)
    group["avg_session_duration"] = group["session_seconds"].rolling("24h").mean().round(3)
    group["vendor_id"] = vendor_name
    return group

df = df.groupby("vendor_id", group_keys=False).apply(rolling_24h)
df = df.reset_index()

# ---- clean up ----
df = df.drop(columns=["is_login", "is_failed_login", "session_seconds", "timestamp_dt"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["session_id"] = [f"SESS-{i+1:08d}" for i in range(len(df))]

final_cols = [
    "timestamp","vendor_id","source_ip","destination_ip","destination_port",
    "device_id","action_type","auth_result","geo_location",
    "flow_duration","flow_bytes","data_volume_mb",
    "total_fwd_packets","total_backward_packets",
    "fwd_packet_length_mean","bwd_packet_length_mean",
    "login_frequency_24h","avg_session_duration","off_hours_flag","failed_login_count",
    "resource_accessed","session_id","_ground_truth_label"
]
df = df[final_cols]

out_path = f"{INPUT_DIR}/full_week_vendor_dataset.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Final shape: {df.shape}")
print("\nGround truth label distribution:")
print(df["_ground_truth_label"].value_counts())