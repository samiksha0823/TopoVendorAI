"""
FastAPI server exposing Gemma-driven incident analysis + raw pipeline
data to the dashboard.

PATH NOTE: this file lives at project_root/api.py — a SIBLING of Output/,
Dataset/, src/, and gemma_agent/. Run it from the project root:

    uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os

from Gemma_agent.agent import SOCAgent

app = FastAPI(title="TopoVendor AI - SOC Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# api.py sits directly inside project_root, so output/ is the actual generated-data folder
# Keep compatibility with historical uppercase folder names as well.
base_dir = os.path.dirname(__file__)
for candidate in ["output", "Output"]:
    path = os.path.join(base_dir, candidate)
    if os.path.isdir(path):
        OUTPUT_DIR = path
        break
else:
    OUTPUT_DIR = os.path.join(base_dir, "output")

agent = SOCAgent(model="gemma3:4b")  # swap to "gemma3:1b" if too slow


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert NaN/NaT/NA values into None so FastAPI can JSON encode the result."""
    if df is None or df.empty:
        return df
    clean = df.copy()
    clean = clean.astype(object).where(pd.notnull(clean), None)
    return clean


def _json_records(df: pd.DataFrame):
    return _sanitize_dataframe(df).to_dict(orient="records")


def _load(name: str) -> pd.DataFrame:
    """
    Loads a CSV from the project output directory and sanitizes it for JSON
    serialization.

    IMPORTANT: pandas represents empty/missing cells as NaN, and standard
    JSON does not support NaN — FastAPI's encoder will throw a 500 Internal
    Server Error if any NaN reaches the response. df.where(pd.notnull(df), None)
    converts all NaN values to None, which serializes cleanly as JSON null.
    """
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{name} not found in {OUTPUT_DIR}")
    df = pd.read_csv(path)
    return _sanitize_dataframe(df)


@app.get("/api/vendors/{vendor_id}/risk")
def get_risk(vendor_id: str):
    df = _load("vendor_dynamic_risk_scores.csv")
    rows = df[df["vendor_id"] == vendor_id].sort_values("window_start")
    if rows.empty:
        raise HTTPException(404, f"No data for vendor {vendor_id}")
    return _json_records(rows)


@app.get("/api/vendors/{vendor_id}/analyze")
def analyze_vendor(vendor_id: str, window_start: str = None):
    """
    Runs the Gemma agent on the vendor's latest (or specified) window.
    This is the live, on-demand SOC assistant call.
    """
    df = _load("vendor_dynamic_risk_scores.csv")
    vdf = df[df["vendor_id"] == vendor_id]
    if vdf.empty:
        raise HTTPException(404, f"No data for vendor {vendor_id}")

    row = (vdf[vdf["window_start"] == window_start].iloc[0]
           if window_start else vdf.sort_values("window_start").iloc[-1])

    clean_row = _sanitize_dataframe(row.to_frame().T).iloc[0].to_dict()
    result = agent.analyze_event(clean_row)
    return {
        "vendor_id": vendor_id,
        "window_start": row["window_start"],
        "risk_tier": row["risk_tier"],
        "dynamic_risk_score": row["dynamic_risk_score"],
        **result,
    }


@app.get("/api/incidents")
def get_all_incidents(min_tier: str = None):
    """Batch view for dashboard — pre-computed or on-the-fly, not re-querying Gemma per row here."""
    df = _load("vendor_dynamic_risk_scores.csv")
    if min_tier:
        normalized = min_tier.strip().title()
        allowed = {"Low", "Medium", "High"}
        if normalized not in allowed:
            raise HTTPException(status_code=400, detail=f"min_tier must be one of {sorted(allowed)}")
        df = df[df["risk_tier"].astype(str).str.title() == normalized]
    return _json_records(df.sort_values("dynamic_risk_score", ascending=False))


@app.post("/api/incidents/batch-analyze")
def batch_analyze(limit: int = 20):
    """
    Runs Gemma over the top N highest-risk rows only (not the whole dataset —
    keeps this usable on a 4GB laptop GPU). Good for demo/offline batch mode.
    """
    df = _load("vendor_dynamic_risk_scores.csv")
    top = df[df["risk_tier"] != "Low"].sort_values("dynamic_risk_score", ascending=False).head(limit)

    results = []
    for _, row in top.iterrows():
        clean_row = _sanitize_dataframe(row.to_frame().T).iloc[0].to_dict()
        res = agent.analyze_event(clean_row)
        results.append({
            "vendor_id": row["vendor_id"],
            "window_start": row["window_start"],
            "risk_tier": row["risk_tier"],
            "dynamic_risk_score": row["dynamic_risk_score"],
            **res,
        })
    return results


@app.get("/health")
def health():
    return {"status": "ok"}