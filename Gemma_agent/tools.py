"""
Tool definitions the agent can call. This is the 'function calling' surface —
Gemma picks a tool by name via JSON, we execute it in Python, and feed the
observation back into the conversation. Includes a lightweight local RAG
retriever (TF-IDF) over a knowledge base of past incidents / MITRE ATT&CK
descriptions — no embedding model needed, so it's cheap on a 4GB card.

PATH NOTE: this file lives at project_root/gemma_agent/tools.py, a SIBLING
of Output/ and Dataset/ (not nested two levels down inside src/ anymore).
So we only go UP ONE level from this file to reach the project root.
"""
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Output")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "Dataset")

# ---------------------------------------------------------------------------
# Tool schemas — exposed to Gemma inside the prompt so it knows what it can call
# ---------------------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "get_vendor_history",
        "description": "Get the last N risk-score windows for a vendor to see the trend over time.",
        "parameters": {"vendor_id": "string", "n_windows": "integer (default 5)"},
    },
    {
        "name": "get_topology_context",
        "description": "Get topological/graph features (degree changes, persistence entropy, betti numbers) for a vendor at a specific window.",
        "parameters": {"vendor_id": "string", "window_start": "string"},
    },
    {
        "name": "search_threat_intel",
        "description": "Search a local knowledge base of MITRE ATT&CK techniques and past incident writeups for context matching given anomaly keywords.",
        "parameters": {"query": "string"},
    },
    {
        "name": "get_blast_radius",
        "description": "Get the count of other vendors/resources sharing access with this vendor (shared_resource_count) to estimate lateral movement risk.",
        "parameters": {"vendor_id": "string", "window_start": "string"},
    },
]


def get_vendor_history(vendor_id: str, n_windows: int = 5) -> dict:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "vendor_dynamic_risk_scores.csv"))
    vdf = df[df["vendor_id"] == vendor_id].sort_values("window_start").tail(n_windows)
    if vdf.empty:
        return {"error": f"No history found for vendor {vendor_id}"}
    return vdf[["window_start", "dynamic_risk_score", "risk_tier", "risk_reasons"]].to_dict(orient="records")


def get_topology_context(vendor_id: str, window_start: str) -> dict:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "vendor_topology_features.csv"))
    row = df[(df["vendor_id"] == vendor_id) & (df["window_start"] == window_start)]
    if row.empty:
        return {"error": "No topology row found for that vendor/window"}
    cols = ["degree", "degree_delta_vs_prev_window", "graph_betti_0", "graph_betti_1",
            "graph_persistence_entropy", "topological_change_score"]
    return row[cols].to_dict(orient="records")[0]


def get_blast_radius(vendor_id: str, window_start: str) -> dict:
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "vendor_dynamic_risk_scores.csv"))
    row = df[(df["vendor_id"] == vendor_id) & (df["window_start"] == window_start)]
    if row.empty:
        return {"error": "No row found"}
    return {
        "shared_resource_count": int(row.iloc[0]["shared_resource_count"]),
        "new_resources_count": int(row.iloc[0]["new_resources_count"]),
        "new_resources_list": row.iloc[0]["new_resources_list"],
    }


# --- Lightweight local RAG over a small knowledge base -----------------------
_KB_CACHE = {"vectorizer": None, "matrix": None, "docs": None}


def _load_kb():
    """
    Expects Dataset/threat_intel_kb.csv with columns: id, title, text
    (e.g. MITRE ATT&CK technique summaries + past incident postmortems).
    Falls back to a tiny built-in KB if the file doesn't exist yet.
    """
    kb_path = os.path.join(DATASET_DIR, "threat_intel_kb.csv")
    if os.path.exists(kb_path):
        docs_df = pd.read_csv(kb_path)
    else:
        docs_df = pd.DataFrame([
            {"id": "T1078", "title": "Valid Accounts",
             "text": "Adversaries use compromised credentials of valid accounts, including vendor/third-party accounts, to bypass access controls and blend in with normal activity."},
            {"id": "T1199", "title": "Trusted Relationship",
             "text": "Adversaries breach organizations through third parties with privileged access, exploiting the trust relationship to move into the target environment."},
            {"id": "T1550", "title": "Use Alternate Authentication Material",
             "text": "Adversaries reuse stolen session tokens or credentials to bypass MFA and normal authentication flows."},
        ])
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs_df["text"])
    _KB_CACHE.update({"vectorizer": vectorizer, "matrix": matrix, "docs": docs_df})


def search_threat_intel(query: str, top_k: int = 2) -> list:
    if _KB_CACHE["vectorizer"] is None:
        _load_kb()
    vec = _KB_CACHE["vectorizer"].transform([query])
    sims = cosine_similarity(vec, _KB_CACHE["matrix"]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    docs = _KB_CACHE["docs"]
    return [
        {"id": docs.iloc[i]["id"], "title": docs.iloc[i]["title"], "text": docs.iloc[i]["text"], "score": float(sims[i])}
        for i in top_idx if sims[i] > 0
    ]


DISPATCH = {
    "get_vendor_history": get_vendor_history,
    "get_topology_context": get_topology_context,
    "search_threat_intel": search_threat_intel,
    "get_blast_radius": get_blast_radius,
}


def call_tool(name: str, args: dict):
    if name not in DISPATCH:
        return {"error": f"Unknown tool: {name}"}
    try:
        return DISPATCH[name](**args)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}