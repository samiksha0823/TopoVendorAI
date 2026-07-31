"""
build_behavioral_anomaly_model.py
-----------------------------------
Phase 4 of the TopoVendor AI pipeline: trains an unsupervised behavioral
anomaly model per vendor and scores every event in the full week.

APPROACH:
  1. Learn each vendor's "normal" baseline from Monday (100% BENIGN day) --
     mean/std of its behavioral features, its normal geo set, its normal
     device set. This mirrors Phase 3 (Vendor Behavior Profiling).
  2. Z-score every event's numeric features against ITS OWN vendor's
     baseline (not a global mean) -- so the model learns "how unusual is
     this FOR THIS VENDOR", not "unusual in general".
  3. Add binary deviation flags: off_hours_flag (already in the data),
     is_unusual_geo, is_new_device (both computed vs. the vendor's
     Monday-baseline sets).
  4. Fit a single IsolationForest on the standardized baseline feature
     space (Monday data across all vendors) -- unsupervised, exactly as
     your TRD specifies (no labels used in training).
  5. Score every event in the full week; calibrate into a 0-1
     behavioral_anomaly_score using the baseline's own score distribution.
  6. Validate against the REAL ground-truth attack labels (same
     methodology used for the TDA/graph script) to show which attack
     types this layer catches -- expected to be strong on volumetric/
     credential attacks (DDoS, brute-force, DoS) where TDA was weak.

FOLDER CONVENTIONS (this pipeline-wide):
    Dataset/    -- holds the full-week generated dataset (input)
    output/     -- holds every phase's output artifacts (created if missing)

INPUT:
    Dataset/full_week_vendor_dataset.csv

OUTPUT:
    output/vendor_behavioral_scores.csv   (per-event behavioral_anomaly_score
                                            + contributing reason flags)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Pipeline-wide folder conventions
# ---------------------------------------------------------------------------
DATASET_DIR = Path("Dataset")
OUTPUT_DIR = Path("output")

NUMERIC_FEATURES = [
    "flow_duration", "data_volume_mb", "total_fwd_packets", "total_backward_packets",
    "fwd_packet_length_mean", "bwd_packet_length_mean",
    "login_frequency_24h", "avg_session_duration", "failed_login_count",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATASET_DIR / "full_week_vendor_dataset.csv"),
                     help="Path to the full-week dataset. Defaults to Dataset/full_week_vendor_dataset.csv")
    ap.add_argument("--output", default=str(OUTPUT_DIR / "vendor_behavioral_scores.csv"),
                     help="Path to write scored output. Defaults to output/vendor_behavioral_scores.csv")
    ap.add_argument("--baseline_day", default="2026-07-01", help="Date (YYYY-MM-DD) used as the pure-normal baseline day (Monday).")
    ap.add_argument("--contamination", type=float, default=0.02,
                     help="Expected proportion of anomalies in the baseline fit -- kept low since baseline day is ~100%% benign.")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find input dataset at '{input_path}'. "
            f"Make sure the generated dataset is placed in the '{DATASET_DIR}/' folder, "
            f"or pass --input explicitly."
        )

    print(f"Loading {input_path} ...")
    usecols = ["timestamp", "vendor_id", "geo_location", "device_id", "off_hours_flag",
               "_ground_truth_label"] + NUMERIC_FEATURES
    dtypes = {c: "float32" for c in NUMERIC_FEATURES}
    dtypes["off_hours_flag"] = "int8"
    df = pd.read_csv(input_path, usecols=usecols, dtype=dtypes)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)  # explicit parse -- handles both
    # the original 'Z'-suffixed format and the '+00:00'-suffixed format some combine scripts write
    print(f"Loaded {len(df):,} rows")

    baseline_mask = df["timestamp"].dt.date.astype(str) == args.baseline_day
    baseline_df = df[baseline_mask]
    print(f"Baseline day ({args.baseline_day}): {len(baseline_df):,} rows, "
          f"{(baseline_df['_ground_truth_label'] != 'BENIGN').sum()} non-benign (should be ~0)")

    # ---- Phase 3-style per-vendor baseline profile ----
    vendor_geo_baseline = {}
    vendor_device_baseline = {}

    vendor_mean = {v: g[NUMERIC_FEATURES].mean() for v, g in baseline_df.groupby("vendor_id")}
    vendor_std = {v: g[NUMERIC_FEATURES].std().replace(0, 1.0).fillna(1.0) for v, g in baseline_df.groupby("vendor_id")}
    for v in df["vendor_id"].unique():
        vendor_geo_baseline[v] = set(baseline_df.loc[baseline_df["vendor_id"] == v, "geo_location"].unique())
        vendor_device_baseline[v] = set(baseline_df.loc[baseline_df["vendor_id"] == v, "device_id"].unique())

    print("Z-scoring every event against its own vendor's baseline ...")
    z_cols = []
    for c in NUMERIC_FEATURES:
        means = df["vendor_id"].map(lambda v: vendor_mean[v][c])
        stds = df["vendor_id"].map(lambda v: vendor_std[v][c])
        zcol = f"z_{c}"
        df[zcol] = ((df[c] - means) / stds).clip(-10, 10).astype("float32")
        z_cols.append(zcol)

    df["is_unusual_geo"] = 0
    df["is_new_device"] = 0

    baseline_geo_pairs = baseline_df[["vendor_id", "geo_location"]].drop_duplicates()
    baseline_geo_pairs["_is_baseline_geo"] = 1
    df = df.merge(baseline_geo_pairs, on=["vendor_id", "geo_location"], how="left")
    df["is_unusual_geo"] = df["_is_baseline_geo"].isna().astype("int8")
    df = df.drop(columns=["_is_baseline_geo"])

    baseline_device_pairs = baseline_df[["vendor_id", "device_id"]].drop_duplicates()
    baseline_device_pairs["_is_baseline_device"] = 1
    df = df.merge(baseline_device_pairs, on=["vendor_id", "device_id"], how="left")
    df["is_new_device"] = df["_is_baseline_device"].isna().astype("int8")
    df = df.drop(columns=["_is_baseline_device"])

    model_features = z_cols + ["off_hours_flag", "is_unusual_geo", "is_new_device"]

    # ---- fit IsolationForest on baseline (normal) data only, UNSUPERVISED (no labels used) ----
    print("Fitting IsolationForest on baseline (normal) data ...")
    X_baseline = df.loc[baseline_mask, model_features].values
    clf = IsolationForest(
        n_estimators=200, contamination=args.contamination,
        random_state=42, n_jobs=-1
    )
    clf.fit(X_baseline)

    # ---- score every event in the full week ----
    print("Scoring all events ...")
    raw_scores = clf.decision_function(df[model_features].values)  # higher = more normal
    # calibrate: invert + min-max normalize using baseline's own score range, so
    # 0 = as normal as typical baseline behavior, 1 = most anomalous seen
    baseline_raw = clf.decision_function(X_baseline)
    score_min = min(raw_scores.min(), baseline_raw.min())
    score_max = max(raw_scores.max(), baseline_raw.max())
    df["behavioral_anomaly_score"] = (1 - (raw_scores - score_min) / (score_max - score_min)).round(4).astype("float32")

    # ---- reason flags for explainability (Phase 4/6 requirement) ----
    df["reason_off_hours"] = df["off_hours_flag"] == 1
    df["reason_unusual_geo"] = df["is_unusual_geo"] == 1
    df["reason_new_device"] = df["is_new_device"] == 1
    df["reason_high_failed_logins"] = df["z_failed_login_count"] > 2
    df["reason_high_data_volume"] = df["z_data_volume_mb"] > 2
    df["reason_unusual_packet_pattern"] = (df["z_total_fwd_packets"].abs() > 3) | (df["z_total_backward_packets"].abs() > 3)

    out_cols = ["timestamp", "vendor_id", "behavioral_anomaly_score",
                "reason_off_hours", "reason_unusual_geo", "reason_new_device",
                "reason_high_failed_logins", "reason_high_data_volume", "reason_unusual_packet_pattern",
                "_ground_truth_label"]
    df[out_cols].to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} scored rows to {output_path}")

    # ---- validation against real ground truth (same style as TDA validation) ----
    print("\n=== VALIDATION vs. real ground-truth labels ===")
    benign_mean = df.loc[df["_ground_truth_label"] == "BENIGN", "behavioral_anomaly_score"].mean()
    print(f"Mean behavioral_anomaly_score, BENIGN rows: {benign_mean:.4f}\n")
    for lbl, g in df[df["_ground_truth_label"] != "BENIGN"].groupby("_ground_truth_label"):
        m = g["behavioral_anomaly_score"].mean()
        print(f"{lbl:30s} n={len(g):7,d}  mean_score={m:.4f}  ratio_vs_benign={m/benign_mean:.2f}x")


if __name__ == "__main__":
    main()