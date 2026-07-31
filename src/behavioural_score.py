"""
visualize_behavioral_scores.py
-----------------------------------
Standalone visualization step for Phase 4 output. Reads
vendor_behavioral_scores.csv (produced by build_behavioral_anomaly_model.py)
and produces a fixed set of diagnostic plots.

PLOTS PRODUCED (all saved as PNGs into --outdir):
  1. score_distribution.png
  2. mean_score_by_label.png
  3. mean_score_by_vendor.png
  4. reason_flag_frequency.png
  5. score_over_time.png
  6. roc_curve.png
  7. precision_recall_curve.png
  8. confusion_matrix.png
  9. anomaly_persistence.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------------------------
# Path setup: resolve defaults relative to THIS script's location, not
# whatever directory the user happens to run python from.
#
#   TopoVendorAI/
#     Output/vendor_behavioral_scores.csv   <- input lives here
#     behavioral_score_plots/               <- plots go here
#     src/behavioural_score.py              <- this file
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from src/

DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "Output", "vendor_behavioral_scores.csv")
DEFAULT_OUTDIR = os.path.join(PROJECT_ROOT, "behavioral_score_plots")

REASON_COLS = [
    "reason_off_hours",
    "reason_unusual_geo",
    "reason_new_device",
    "reason_high_failed_logins",
    "reason_high_data_volume",
    "reason_unusual_packet_pattern",
]

BENIGN_COLOR = "#79de21"
ATTACK_COLOR = "#ff1f1f"


def load(input_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find input CSV at: {input_path}\n"
            f"  -> Pass the correct path explicitly with --input, e.g.\n"
            f"     python behavioural_score.py --input \"E:\\TopoVendorAI\\Output\\vendor_behavioral_scores.csv\""
        )
    df = pd.read_csv(input_path)

    if "timestamp" not in df.columns:
        raise ValueError(f"Expected a 'timestamp' column in {input_path}, found: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        print(f"  WARNING: {n_bad:,} rows had unparseable timestamps and were dropped.")
        df = df.dropna(subset=["timestamp"])

    for c in REASON_COLS:
        if c in df.columns:
            df[c] = df[c].astype(bool)
    df["_is_attack"] = df["_ground_truth_label"] != "BENIGN"
    return df


def plot_score_distribution(df, outdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(df.loc[~df["_is_attack"], "behavioral_anomaly_score"], bins=bins,
            alpha=0.6, label="BENIGN", color=BENIGN_COLOR, density=True)
    ax.hist(df.loc[df["_is_attack"], "behavioral_anomaly_score"], bins=bins,
            alpha=0.6, label="ATTACK (all types)", color=ATTACK_COLOR, density=True)
    ax.set_xlabel("behavioral_anomaly_score")
    ax.set_ylabel("density")
    ax.set_title("Behavioral Anomaly Score Distribution: BENIGN vs ATTACK")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "score_distribution.png"), dpi=150)
    plt.close(fig)


def plot_mean_score_by_label(df, outdir):
    means = df.groupby("_ground_truth_label")["behavioral_anomaly_score"].mean().sort_values()
    colors = [BENIGN_COLOR if lbl == "BENIGN" else ATTACK_COLOR for lbl in means.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(means.index, means.values, color=colors)
    ax.set_xlabel("mean behavioral_anomaly_score")
    ax.set_title("Mean Anomaly Score by Ground-Truth Label")
    for i, v in enumerate(means.values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mean_score_by_label.png"), dpi=150)
    plt.close(fig)


def plot_mean_score_by_vendor(df, outdir):
    means = df.groupby("vendor_id")["behavioral_anomaly_score"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(means))))
    ax.barh(means.index.astype(str), means.values, color="#eff702")
    ax.set_xlabel("mean behavioral_anomaly_score")
    ax.set_title("Mean Anomaly Score by Vendor")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "mean_score_by_vendor.png"), dpi=150)
    plt.close(fig)


def plot_reason_flag_frequency(df, outdir):
    present = [c for c in REASON_COLS if c in df.columns]
    benign_rates = df.loc[~df["_is_attack"], present].mean()
    attack_rates = df.loc[df["_is_attack"], present].mean()

    x = np.arange(len(present))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, benign_rates.values, width, label="BENIGN", color=BENIGN_COLOR)
    ax.bar(x + width / 2, attack_rates.values, width, label="ATTACK", color=ATTACK_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("reason_", "") for c in present], rotation=30, ha="right")
    ax.set_ylabel("fraction of rows flagged")
    ax.set_title("Reason Flag Firing Rate: BENIGN vs ATTACK")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "reason_flag_frequency.png"), dpi=150)
    plt.close(fig)


def plot_score_over_time(df, outdir, max_points=50000):
    d = df.sort_values("timestamp")
    if len(d) > max_points:
        d = d.sample(max_points, random_state=42).sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(d.loc[~d["_is_attack"], "timestamp"], d.loc[~d["_is_attack"], "behavioral_anomaly_score"],
               s=4, alpha=0.3, color=BENIGN_COLOR, label="BENIGN")
    ax.scatter(d.loc[d["_is_attack"], "timestamp"], d.loc[d["_is_attack"], "behavioral_anomaly_score"],
               s=6, alpha=0.6, color=ATTACK_COLOR, label="ATTACK")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("behavioral_anomaly_score")
    ax.set_title("Anomaly Score Across the Week")
    ax.legend(markerscale=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "score_over_time.png"), dpi=150)
    plt.close(fig)


def plot_roc_curve(df, outdir):
    fpr, tpr, _ = roc_curve(df["_is_attack"], df["behavioral_anomaly_score"])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=ATTACK_COLOR, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_curve.png"), dpi=150)
    plt.close(fig)


def plot_pr_curve(df, outdir):
    precision, recall, _ = precision_recall_curve(df["_is_attack"], df["behavioral_anomaly_score"])
    ap = average_precision_score(df["_is_attack"], df["behavioral_anomaly_score"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color=BENIGN_COLOR, lw=2, label=f"PR curve (AP = {ap:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "precision_recall_curve.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix_threshold(df, outdir, threshold):
    preds = (df["behavioral_anomaly_score"] >= threshold).astype(int)
    cm = confusion_matrix(df["_is_attack"].astype(int), preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"])
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title(f"Confusion Matrix (Threshold = {threshold})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)


def plot_anomaly_persistence(df, outdir, threshold):
    # Determine which events exceed the anomaly threshold
    df["_is_flagged"] = df["behavioral_anomaly_score"] >= threshold

    # Truncate timestamps to the nearest hour to count distinct anomalous periods
    df["hour"] = df["timestamp"].dt.floor("h")

    # Group by vendor and count unique hours where they were flagged
    persistence = df[df["_is_flagged"]].groupby("vendor_id")["hour"].nunique()

    # Sort and take top 20 for readability
    persistence = persistence.sort_values(ascending=False).head(20)

    if persistence.empty:
        print(f"  No vendors persisted above threshold {threshold}. Skipping persistence plot.")
        return

    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(persistence))))
    ax.barh(persistence.index.astype(str), persistence.values, color="#21dece")
    ax.set_xlabel("Distinct Hours Flagged")
    ax.set_ylabel("Vendor ID")
    ax.set_title(f"Anomaly Persistence: Top 20 Vendors (Score >= {threshold})")
    ax.invert_yaxis()  # Highest persistence at the top
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "anomaly_persistence.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT,
                     help=f"Path to vendor_behavioral_scores.csv (default: {DEFAULT_INPUT})")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                     help=f"Directory to save plots into (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="Score cutoff for confusion matrix and persistence plotting")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Loading {args.input} ...")
    df = load(args.input)
    print(f"Loaded {len(df):,} rows")

    print("Plotting score distribution ...")
    plot_score_distribution(df, args.outdir)

    print("Plotting mean score by label ...")
    plot_mean_score_by_label(df, args.outdir)

    print("Plotting mean score by vendor ...")
    plot_mean_score_by_vendor(df, args.outdir)

    print("Plotting reason flag frequency ...")
    plot_reason_flag_frequency(df, args.outdir)

    print("Plotting score over time ...")
    plot_score_over_time(df, args.outdir)

    print("Plotting ROC curve ...")
    plot_roc_curve(df, args.outdir)

    print("Plotting Precision-Recall curve ...")
    plot_pr_curve(df, args.outdir)

    print(f"Plotting confusion matrix at threshold {args.threshold} ...")
    plot_confusion_matrix_threshold(df, args.outdir, args.threshold)

    print(f"Plotting anomaly persistence at threshold {args.threshold} ...")
    plot_anomaly_persistence(df, args.outdir, args.threshold)

    print(f"\nSaved 9 plots to {args.outdir}/")


if __name__ == "__main__":
    main()