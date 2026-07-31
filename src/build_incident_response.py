"""
build_incident_response.py
-----------------------------
Phase 7 of the TopoVendor AI pipeline: rule-based incident response
recommendation engine. Takes the fused dynamic risk score + explainability
reasons from Phase 6 and maps them to concrete, actionable SOC responses.

DESIGN:
Risk tier sets the BASE severity of response; the specific reason flags
(carried through from Phase 4/5 explainability) determine which SPECIFIC
actions are recommended -- so two "High" risk vendors can get different
action lists depending on WHY they're high (e.g. credential-stuffing
pattern -> enforce MFA + block, vs. new-resource/lateral-movement pattern
-> restrict access + revoke sessions), matching your PRD's Phase 7 spec:
  - Enable MFA / block-suspend vendor access
  - Revoke active sessions
  - Restrict access to sensitive resources
  - Notify SOC
  - Initiate detailed investigation

This is intentionally a transparent, human-auditable RULE ENGINE (not
another ML model) -- appropriate for a phase whose whole point is
explainable, actionable, human-in-the-loop response.

FOLDER CONVENTIONS (this pipeline-wide):
    Dataset/    -- holds the full-week generated dataset
    output/     -- holds every phase's output artifacts (created if missing);
                   this phase reads the Phase 6 output from here too.

INPUT:
    output/vendor_dynamic_risk_scores.csv   (Phase 6 output)

OUTPUT:
    output/vendor_incident_response.csv     (adds recommended_actions,
                                              action_priority per vendor-window)
"""

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Pipeline-wide folder conventions
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")


def recommend_actions(row):
    tier = row["risk_tier"]
    reasons = row["risk_reasons"]
    actions = []

    if tier == "Low":
        actions.append("No action required — continue standard monitoring")
        return "; ".join(actions), "Routine"

    # --- Medium: analyst awareness + light-touch containment, no hard block yet ---
    if tier == "Medium":
        actions.append("Flag for SOC analyst review")
        if "failed login" in reasons.lower():
            actions.append("Prompt vendor to re-authenticate with MFA")
        if "unfamiliar location" in reasons.lower() or "unrecognized device" in reasons.lower():
            actions.append("Increase monitoring frequency for this vendor")
        if "resource(s) never seen before" in reasons.lower() or "structural change" in reasons.lower():
            actions.append("Log and review newly accessed resources")
        if len(actions) == 1:
            actions.append("Increase monitoring frequency for this vendor")
        return "; ".join(actions), "Elevated"

    # --- High: hard containment actions, always escalate to SOC ---
    actions.append("Notify Security Operations Center (SOC)")
    actions.append("Initiate detailed security investigation")

    if "failed login" in reasons.lower():
        actions.append("Enforce MFA on next login attempt")
        actions.append("Consider temporary account lockout after further failed attempts")
    if "unfamiliar location" in reasons.lower() or "unrecognized device" in reasons.lower():
        actions.append("Revoke active sessions")
        actions.append("Require re-verification of vendor identity")
    if "resource(s) never seen before" in reasons.lower() or "sudden change in number of resources" in reasons.lower() or "structural change" in reasons.lower():
        actions.append("Restrict access to sensitive resources pending review")
    if "abnormally high data transfer" in reasons.lower():
        actions.append("Suspend data export capability for this vendor")
    if "unusual network traffic pattern" in reasons.lower():
        actions.append("Block/suspend vendor network access pending investigation")

    # ensure at least one hard containment action always accompanies a High tier
    hard_actions = {"Revoke active sessions", "Restrict access to sensitive resources pending review",
                    "Block/suspend vendor network access pending investigation",
                    "Suspend data export capability for this vendor"}
    if not (set(actions) & hard_actions):
        actions.append("Restrict access to sensitive resources pending review")

    return "; ".join(actions), "Critical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(OUTPUT_DIR / "vendor_dynamic_risk_scores.csv"),
                     help="Path to Phase 6 output. Defaults to output/vendor_dynamic_risk_scores.csv")
    ap.add_argument("--output", default=str(OUTPUT_DIR / "vendor_incident_response.csv"),
                     help="Path to write final output. Defaults to output/vendor_incident_response.csv")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Could not find Phase 6 risk scores at '{input_path}'. "
            f"Run build_risk_fusion.py first, or pass --input explicitly."
        )

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} vendor-window risk rows from {input_path}")

    results = df.apply(recommend_actions, axis=1)
    df["recommended_actions"] = results.apply(lambda x: x[0])
    df["action_priority"] = results.apply(lambda x: x[1])

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")

    print("\n=== Action priority distribution ===")
    print(df["action_priority"].value_counts())

    print("\n=== Sample High-tier recommendations (first 5) ===")
    high = df[df["risk_tier"] == "High"][["window_start", "vendor_id", "dynamic_risk_score", "risk_reasons", "recommended_actions", "vendor_attack_types_this_window"]]
    for _, r in high.head(5).iterrows():
        print(f"\n[{r['window_start']}] {r['vendor_id']}  risk={r['dynamic_risk_score']:.3f}  (real attack: {r['vendor_attack_types_this_window'] or 'none'})")
        print(f"  Reasons: {r['risk_reasons']}")
        print(f"  Actions: {r['recommended_actions']}")

    print("\n=== Validation: does 'Critical' priority concentrate on real attacks? ===")
    crit = df[df["action_priority"] == "Critical"]
    print(f"Critical-priority windows: {len(crit)}, of which real attack present: {crit['vendor_has_attack_this_window'].sum()} ({crit['vendor_has_attack_this_window'].mean()*100:.1f}%)")


if __name__ == "__main__":
    main()