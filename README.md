# TopoVendor AI

**Third-Party Vendor Risk Detection & Supply Chain Attack Prevention**

An AI-powered platform that continuously assesses third-party vendor security posture, detects suspicious activity originating from vendor access, identifies compromised vendor accounts, and surfaces real-time risk scores with incident response recommendations — helping organizations catch supply chain risk before attackers can escalate into critical systems.

---

## Problem Statement

Vendors and third parties are one of the most common entry points for supply chain attacks. A compromised vendor credential, an unusual access pattern, or a sudden expansion into new resources can be the first sign of an attack unfolding — but these signals are easy to miss when buried across raw access logs.

TopoVendor AI continuously scores every vendor's behavior across two independent lenses — **behavioral anomaly detection** and **topological/graph structure analysis** — fuses them into a single dynamic risk score, and flags the vendors that need attention before an incident escalates.

---

## How It Works

The platform runs as a multi-phase pipeline, each phase building on the output of the last:

```
Raw Events (Dataset/)
        │
        ▼
┌─────────────────────────────┐
│ 1. Behavioral Anomaly Model  │  → per-event anomaly scores from vendor
│    (unsupervised)            │    access patterns (time of day, location,
│                               │    device, volume, resource diversity)
└──────────────┬───────────────┘
               ▼
┌─────────────────────────────┐
│ 2. Topological Data Analysis │  → graph-based structural features
│    (vendor access graph)     │    (degree changes, persistence entropy,
│                               │    betti numbers, blast radius)
└──────────────┬───────────────┘
               ▼
┌─────────────────────────────┐
│ 3. Risk Fusion               │  → combines behavioral + topological
│                               │    signals into one dynamic_risk_score
│                               │    per vendor per time window
└──────────────┬───────────────┘
               ▼
┌─────────────────────────────┐
│ 4. Incident Response         │  → maps risk tier + reasons to
│    (rule-based baseline)     │    recommended containment actions
│                               │    and a priority level
└──────────────┬───────────────┘
               ▼
     Output/*.csv
               │
               ▼
┌─────────────────────────────┐
│ 5. Gemma 3 SOC Agent         │  → for Medium/High risk vendors, a local
│    (gemma_agent/)            │    Gemma 3 agent reasons over the event,
│                               │    calls tools to pull vendor history,
│                               │    topology context, and threat intel,
│                               │    then generates a structured incident
│                               │    summary + prioritized containment plan
└──────────────┬───────────────┘
               ▼
        api.py (FastAPI)  →  Dashboard
```

Low-risk vendors skip the Gemma call entirely (a fast rule-based path returns "continue standard monitoring") — the agent is only invoked where it adds value, which keeps the platform usable on modest hardware.

Each vendor, in each time window, ends up with a **dynamic risk score**, a **risk tier** (Low / Medium / High), a list of **specific anomaly reasons**, and **recommended containment actions** — visualized live on the dashboard.

---

## Project Structure

```
TopoVendorAI/
│
├── dashboard/                 # Frontend dashboard (visual risk command center)
├── Dataset/                   # Input CSVs — raw vendor access/event data
├── Output/                    # Pipeline output CSVs (generated)
│
├── src/
│   ├── build_behavioral_anomaly_model.py   # Phase 1
│   ├── build_vendor_graph_tda.py           # Phase 2
│   ├── build_risk_fusion.py                # Phase 3
│   ├── build_incident_response.py          # Phase 4
│   └── ... (other pipeline scripts)
│
├── gemma_agent/                # Phase 5 — Gemma 3 SOC assistant (agent, tools, prompts)
│   ├── __init__.py
│   ├── client.py               # Ollama client wrapper
│   ├── tools.py                # Function-calling tools + local RAG over threat intel
│   ├── prompts.py               # System prompt + structured output schema
│   └── agent.py                # ReAct-style multi-step agent loop
│
└── api.py                      # FastAPI server exposing pipeline data + Gemma agent to the dashboard
```

---

## Output Files

| File | Description |
|---|---|
| `Output/vendor_dynamic_risk_scores.csv` | Core scoring output — behavioral + topological components fused into `dynamic_risk_score`, `risk_tier`, and `risk_reasons` per vendor per window |
| `Output/vendor_topology_features.csv` | Raw graph/topological features per vendor per window — degree, betti numbers, persistence entropy, blast radius |
| `Output/vendor_incident_response.csv` | Risk scores enriched with `recommended_actions` and `action_priority` |

Key columns in `vendor_dynamic_risk_scores.csv`:

- `vendor_id`, `window_start` — identifies the vendor and time window
- `behavioral_component`, `topological_component` — the two underlying signal sources
- `dynamic_risk_score` — fused 0–1 risk score
- `risk_tier` — Low / Medium / High
- `risk_reasons` — human-readable list of what triggered the score
- `new_resources_count`, `new_resources_list` — resources accessed for the first time by this vendor
- `shared_resource_count` — how many other vendors share access to the same resources (blast radius indicator)
- `graph_persistence_entropy` — topological stability measure; sudden drops often indicate bypassed network segmentation
- `vendor_has_attack_this_window`, `vendor_attack_types_this_window` — ground-truth labels (present in validation/simulated data)

---

## Setup

### Prerequisites

- Python 3.9+
- pip

### Install dependencies

```bash
pip install pandas numpy scikit-learn networkx
```

(Add any additional libraries your specific pipeline scripts import — e.g. `gudhi` or `ripser` if used for the TDA phase, `plotly`/`streamlit` if using the Python dashboard variant.)

### Run the pipeline

Run each phase in order from the `src/` directory — each script depends on the previous one's output:

```bash
cd src

# Phase 1 — behavioral anomaly detection
python build_behavioral_anomaly_model.py

# Phase 2 — topological graph analysis
python build_vendor_graph_tda.py

# Phase 3 — risk fusion
python build_risk_fusion.py

# Phase 4 — incident response mapping
python build_incident_response.py
```

Confirm the output was generated:

```bash
python -c "import os; print(os.listdir('../Output'))"
```

You should see `vendor_dynamic_risk_scores.csv`, `vendor_topology_features.csv`, and `vendor_incident_response.csv`.

> **Path note:** scripts and downstream tools expect the output folder to be named `Output/` (capital O) at the project root. Keep casing consistent across all scripts to avoid silent path mismatches.

---

## AI Layer — Gemma 3 SOC Agent

Once the pipeline has produced `Output/vendor_dynamic_risk_scores.csv`, the `gemma_agent/` module and `api.py` turn those static scores into live, explainable incident analysis using a locally-run **Gemma 3** model.

### What it does

For any vendor at Medium or High risk, the agent:

1. Reasons over the risk event (score, tier, anomaly reasons, new resources accessed)
2. Optionally calls tools to gather more context — vendor history trend, topological graph features, blast-radius/shared-resource data, and a local threat-intel knowledge base (MITRE ATT&CK-style entries) via lightweight RAG
3. Produces a structured, schema-constrained output: a plain-English incident summary, a prioritized list of containment actions, and a priority level (Routine / Elevated / Critical)

Low-risk vendors skip the model call entirely — the agent focuses compute where it actually matters.

### 1. Install Ollama

Ollama runs Gemma 3 locally and serves it over a REST API.

- **Windows / macOS:** download from https://ollama.com/download
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Verify:
```bash
ollama --version
```

### 2. Start the Ollama service

```bash
ollama serve
```
Leave this running — it serves the model on `http://localhost:11434`. (Often starts automatically after install; check for the tray icon on Windows/Mac.)

### 3. Pull the Gemma 3 model

```bash
ollama pull gemma3:4b
ollama pull gemma3:1b   # lighter fallback for lower-VRAM machines
```

`gemma3:4b` is the default in `gemma_agent/agent.py` — swap to `gemma3:1b` there if inference feels slow on your hardware (roughly 4GB VRAM or less).

### 4. Install Python dependencies for the agent + API

```bash
pip install fastapi uvicorn requests scikit-learn
```

### 5. Start the API server

From the project root (where `api.py` lives):

```bash
uvicorn api:app --reload --port 8000
```

Verify it's up:
```
http://localhost:8000/health   →  {"status": "ok"}
```

Key endpoints:

| Endpoint | Description |
|---|---|
| `GET /api/incidents` | All risk-scored vendor windows (optionally filter with `?min_tier=Medium`) |
| `GET /api/vendors/{vendor_id}/risk` | Full risk history for one vendor |
| `GET /api/vendors/{vendor_id}/analyze` | Runs the live Gemma agent on that vendor's latest (or specified) window |
| `POST /api/incidents/batch-analyze?limit=N` | Runs the Gemma agent over the top N highest-risk vendors |

### 6. Optional: local threat-intel knowledge base

The agent's `search_threat_intel` tool looks for `Dataset/threat_intel_kb.csv` (columns: `id`, `title`, `text`) to ground its analysis in MITRE ATT&CK-style context. If the file isn't present, it falls back to a small built-in set of entries — add your own for richer, more specific reasoning.

### Common issues

| Problem | Fix |
|---|---|
| Model responses are slow | Switch to `gemma3:1b` in `SOCAgent(model=...)` inside `gemma_agent/agent.py` |
| `Connection refused` on port 11434 | `ollama serve` isn't running |
| API can't reach Ollama | Start Ollama before starting `uvicorn` |
| `CUDA out of memory` | Close other GPU-heavy apps, or use `gemma3:1b` |

---

## Dashboard

The `dashboard/` folder contains a visual risk command center that pulls live data from `api.py`:

- **Vendor Risk Radar** — a radial view of every vendor's current risk score, colored by tier
- **Vendor grid** — quick-glance cards per vendor with score and tier
- **Live Alert Feed** — chronological feed of Elevated/Critical risk windows across all vendors
- **Vendor detail view** — risk trend sparkline, detected anomaly reasons, and a live **Gemma SOC Agent** panel — click to run the agent on demand and see its containment recommendations appear in real time

### Running the dashboard

Make sure both of these are running first:
```bash
ollama serve                              # Terminal 1
uvicorn api:app --reload --port 8000      # Terminal 2, from project root
```

Then open `dashboard/index.html` directly in a browser — it's a static file that talks to the API over `fetch()`, no build step required.

### Full run order, start to finish

```bash
# 1. Start Ollama (keep running)
ollama serve

# 2. Regenerate pipeline data (only needed if Dataset/ changed)
cd src
python build_behavioral_anomaly_model.py
python build_vendor_graph_tda.py
python build_risk_fusion.py
python build_incident_response.py
cd ..

# 3. Start the API (keep running)
uvicorn api:app --reload --port 8000

# 4. Open dashboard/index.html in a browser
```

---

## Team Synorix
