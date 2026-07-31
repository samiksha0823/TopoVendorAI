"""
build_risk_fusion.py
----------------------
Phase 6 of the TopoVendor AI pipeline: fuses the Phase 4 behavioral
anomaly score with the Phase 5 topological change score into a single
dynamic vendor risk score, categorized Low / Medium / High, with
human-readable explainability reasons -- feeding directly into Phase 7
(incident response recommendations).

FUSION LOGIC:
Phase 4 and Phase 5 validated as catching DIFFERENT, mostly non-overlapping
attack types (behavioral AI: DoS/DDoS/Heartbleed/credential attacks;
TDA/graph: PortScan/Infiltration/lateral movement). A plain weighted
AVERAGE would dilute an attack that only ONE layer sees strongly (e.g.
PortScan scores 4.65x on TDA but 0.47x on behavioral -- averaging would
cut that signal roughly in half). Instead we combine them as a
probabilistic OR (noisy-OR):

    fused_risk = 1 - (1 - behavioral_component) * (1 - topological_component)

This means EITHER detector firing strongly is enough to raise the fused
score -- matching the actual product thesis ("AI + TDA together catch
more than either alone") rather than averaging their disagreement away.

FOLDER CONVENTIONS (this pipeline-wide):
    Dataset/    -- holds the full-week generated dataset
    output/     -- holds every phase's output artifacts (created if missing);
                   this phase reads Phase 4 + Phase 5 outputs from here too.

INPUT:
    output/vendor_behavioral_scores.csv   (event-level, from Phase 4)
    output/vendor_topology_features.csv   (window-level, from Phase 5)

OUTPUT:
    output/vendor_dynamic_risk_scores.csv  (one row per vendor per hourly window)
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Pipeline-wide folder conventions
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioral", default=str(OUTPUT_DIR / "vendor_behavioral_scores.csv"),
                     help="Path to Phase 4 output. Defaults to output/vendor_behavioral_scores.csv")
    ap.add_argument("--topology", default=str(OUTPUT_DIR / "vendor_topology_features.csv"),
                     help="Path to Phase 5 output. Defaults to output/vendor_topology_features.csv")
    ap.add_argument("--output", default=str(OUTPUT_DIR / "vendor_dynamic_risk_scores.csv"),
                     help="Path to write fused output. Defaults to output/vendor_dynamic_risk_scores.csv")
    ap.add_argument("--window_minutes", type=int, default=60)
    ap.add_argument("--low_threshold", type=float, default=0.35)
    ap.add_argument("--high_threshold", type=float, default=0.65)
    ap.add_argument("--behavioral_agg", choices=["mean", "max"], default="mean",
                     help="Aggregate event-level behavioral scores per window using mean (smoother, may dilute short bursts) or max (captures worst single event in the window).")
    args = ap.parse_args()

    behavioral_path = Path(args.behavioral)
    topology_path = Path(args.topology)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for p, label in [(behavioral_path, "Phase 4 behavioral scores"), (topology_path, "Phase 5 topology features")]:
        if not p.exists():
            raise FileNotFoundError(
                f"Could not find {label} at '{p}'. Run the earlier phase first, "
                f"or pass the correct path explicitly."
            )

    print(f"Loading behavioral scores (event-level) from {behavioral_path} ...")
    beh = pd.read_csv(behavioral_path)
    beh["timestamp"] = pd.to_datetime(beh["timestamp"], format="mixed", utc=True)
    beh["window_start"] = beh["timestamp"].dt.floor(f"{args.window_minutes}min")

    print("Aggregating behavioral scores to (window, vendor) level ...")
    beh_agg = beh.groupby(["window_start", "vendor_id"]).agg(
        behavioral_score_mean=("behavioral_anomaly_score", "mean"),
        behavioral_score_max=("behavioral_anomaly_score", "max"),
        reason_off_hours=("reason_off_hours", "any"),
        reason_unusual_geo=("reason_unusual_geo", "any"),
        reason_new_device=("reason_new_device", "any"),
        reason_high_failed_logins=("reason_high_failed_logins", "any"),
        reason_high_data_volume=("reason_high_data_volume", "any"),
        reason_unusual_packet_pattern=("reason_unusual_packet_pattern", "any"),
        vendor_has_attack_this_window=("_ground_truth_label", lambda s: (s != "BENIGN").any()),
        vendor_attack_types_this_window=("_ground_truth_label", lambda s: ";".join(sorted(set(s[s != "BENIGN"])))),
    ).reset_index()

    print(f"Loading topology features (already window-level) from {topology_path} ...")
    topo = pd.read_csv(topology_path)
    topo["window_start"] = pd.to_datetime(topo["window_start"], format="mixed", utc=True)

    print("Merging behavioral + topology on (window_start, vendor_id) ...")
    fused = beh_agg.merge(
        topo[["window_start", "vendor_id", "new_resources_count", "new_resources_list",
              "degree_delta_vs_prev_window", "shared_resource_count",
              "graph_persistence_entropy", "topological_change_score"]],
        on=["window_start", "vendor_id"], how="inner"
    )

    # ---- normalize each component into a 0-1 "probability of anomaly" ----
    # behavioral_score_{mean,max} is already 0-1 (calibrated in Phase 4)
    fused["behavioral_component"] = fused[f"behavioral_score_{args.behavioral_agg}"].clip(0, 1)

    # topological_change_score has no fixed upper bound -- min-max normalize
    # using the observed range across this dataset
    t_min, t_max = fused["topological_change_score"].min(), fused["topological_change_score"].max()
    fused["topological_component"] = ((fused["topological_change_score"] - t_min) / (t_max - t_min)).clip(0, 1)

    # ---- noisy-OR fusion ----
    fused["dynamic_risk_score"] = (
        1 - (1 - fused["behavioral_component"]) * (1 - fused["topological_component"])
    ).round(4)

    # ---- risk tier ----
    def tier(score):
        if score >= args.high_threshold:
            return "High"
        if score >= args.low_threshold:
            return "Medium"
        return "Low"
    fused["risk_tier"] = fused["dynamic_risk_score"].apply(tier)

    # ---- explainability: build a human-readable reason list per row ----
    def build_reasons(r):
        reasons = []
        if r["reason_high_failed_logins"]:
            reasons.append("Multiple failed login attempts")
        if r["reason_unusual_geo"]:
            reasons.append("Access from an unfamiliar location")
        if r["reason_new_device"]:
            reasons.append("Access from an unrecognized device")
        if r["reason_off_hours"]:
            reasons.append("Activity outside normal working hours")
        if r["reason_high_data_volume"]:
            reasons.append("Abnormally high data transfer volume")
        if r["reason_unusual_packet_pattern"]:
            reasons.append("Unusual network traffic pattern")
        if r["new_resources_count"] > 0:
            reasons.append(f"Accessed {int(r['new_resources_count'])} resource(s) never seen before for this vendor")
        if abs(r["degree_delta_vs_prev_window"]) >= 3:
            reasons.append("Sudden change in number of resources accessed")
        if r["graph_persistence_entropy"] < 0.3:
            reasons.append("Significant structural change in vendor's interaction topology")
        return "; ".join(reasons) if reasons else "No significant deviations detected"

    fused["risk_reasons"] = fused.apply(build_reasons, axis=1)

    out_cols = [
        "window_start", "vendor_id", "behavioral_component", "topological_component",
        "dynamic_risk_score", "risk_tier", "risk_reasons",
        "new_resources_count", "new_resources_list", "shared_resource_count",
        "graph_persistence_entropy", "vendor_has_attack_this_window", "vendor_attack_types_this_window",
    ]
    fused[out_cols].sort_values(["vendor_id", "window_start"]).to_csv(output_path, index=False)
    print(f"\nSaved {len(fused):,} rows to {output_path}")

    # ---- validation vs. real ground truth ----
    print("\n=== RISK TIER DISTRIBUTION ===")
    print(fused["risk_tier"].value_counts())

    print("\n=== VALIDATION: risk tier vs. real attack presence ===")
    ct = pd.crosstab(fused["risk_tier"], fused["vendor_has_attack_this_window"])
    print(ct)

    attacked = fused[fused["vendor_has_attack_this_window"]]
    normal = fused[~fused["vendor_has_attack_this_window"]]
    high_recall = (attacked["risk_tier"] == "High").mean()
    med_or_high_recall = attacked["risk_tier"].isin(["Medium", "High"]).mean()
    false_high_rate = (normal["risk_tier"] == "High").mean()
    print(f"\nOf all vendor-windows with a REAL attack:")
    print(f"  -> flagged High risk:          {high_recall*100:.1f}%")
    print(f"  -> flagged Medium or High risk: {med_or_high_recall*100:.1f}%")
    print(f"Of all vendor-windows with NO attack (should be low):")
    print(f"  -> falsely flagged High risk:  {false_high_rate*100:.1f}%")

    print("\n=== Mean dynamic_risk_score by attack type ===")
    benign_mean = normal["dynamic_risk_score"].mean()
    print(f"Baseline (no attack): {benign_mean:.4f}")
    for lbl, g in attacked.groupby("vendor_attack_types_this_window"):
        if lbl == "":
            continue
        print(f"{lbl:40s} n={len(g):3d}  mean_risk={g['dynamic_risk_score'].mean():.4f}  ratio={g['dynamic_risk_score'].mean()/benign_mean:.2f}x  High%={(g['risk_tier']=='High').mean()*100:.0f}%")


if __name__ == "__main__":
    main()