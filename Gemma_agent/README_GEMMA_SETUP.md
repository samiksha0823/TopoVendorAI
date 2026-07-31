# Running Gemma 3 Locally (First-Time Setup)

This guide gets Gemma 3 running locally via Ollama, so the `gemma_agent` module can talk to it. Written for teammates who've never run a local LLM before.

---

## 1. Prerequisites

- Windows / macOS / Linux laptop
- ~6 GB free disk space (model weights)
- NVIDIA GPU recommended (not required — Gemma 3 1B/4B run on CPU too, just slower)
- For our team: RTX 3050 4GB VRAM is enough for `gemma3:4b` (quantized) or `gemma3:1b`

Check your GPU/driver first (Windows/Linux with NVIDIA):

```bash
nvidia-smi
```

You should see your GPU listed with driver + CUDA version. If this command isn't found, install/update NVIDIA drivers before continuing.

---

## 2. Install Ollama

Ollama is the local runtime that downloads, quantizes, and serves Gemma models via a simple REST API.

- **Windows:** Download and run the installer from https://ollama.com/download
- **macOS:** Download from https://ollama.com/download, or `brew install ollama`
- **Linux:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

Verify it installed:

```bash
ollama --version
```

---

## 3. Start the Ollama service

Usually starts automatically after install (check for the Ollama icon in your system tray on Windows/Mac). If not running:

```bash
ollama serve
```

Leave this running in a terminal — it's a local server on `http://localhost:11434`.

---

## 4. Pull the Gemma 3 model

For our GPU budget (4GB VRAM), pull the smaller variants:

```bash
ollama pull gemma3:4b
ollama pull gemma3:1b
```

- `gemma3:4b` — better quality, use this by default
- `gemma3:1b` — fallback if `4b` feels slow or VRAM is tight

This downloads ~2.5–3.5 GB per model. Grab coffee.

---

## 5. Sanity-check it works

```bash
ollama run gemma3:4b
```

Type a test prompt like `Explain supply chain attacks in one sentence.` and confirm you get a response. Exit with `/bye` or Ctrl+C.

Check GPU usage while it's generating (separate terminal):

```bash
nvidia-smi
```

You should see `ollama` in the process list using VRAM. If it's using 0% GPU and running purely on CPU, see Troubleshooting below.

---

## 6. Test the API directly

This is what our Python code actually talks to — confirm it responds before touching the agent code:

```bash
curl http://localhost:11434/api/generate -d "{\"model\": \"gemma3:4b\", \"prompt\": \"Say hello in 5 words.\", \"stream\": false}"
```

You should get back a JSON blob with a `"response"` field.

---

## 7. Install Python dependencies for our project

```bash
pip install fastapi uvicorn pandas scikit-learn requests
```

---

## 8. Run our project's Gemma agent

From the project root:

```bash
cd src
uvicorn api:app --reload --port 8000
```

Then test an endpoint (in a browser or curl):

```
http://localhost:8000/health
```

Should return `{"status": "ok"}`. Then try:

```
http://localhost:8000/api/incidents
```

---

## 9. Common issues

| Problem | Fix |
|---|---|
| `ollama: command not found` | Restart terminal after install, or re-run installer |
| Model runs but very slow (>30s/response) | Switch to `gemma3:1b` in `SOCAgent(model="gemma3:1b")` |
| `CUDA out of memory` | Close other GPU apps (browser hardware accel, games); use `gemma3:1b` |
| Ollama using CPU not GPU | Update NVIDIA drivers; confirm `nvidia-smi` shows correct driver version; restart `ollama serve` |
| `Connection refused` on port 11434 | `ollama serve` isn't running — start it in a terminal and leave it open |
| FastAPI can't reach Ollama | Confirm Ollama is running BEFORE starting `uvicorn` |

---

## 10. Quick model swap reference

Change the model anywhere in code by editing:

```python
agent = SOCAgent(model="gemma3:4b")   # or "gemma3:1b"
```

No re-download needed if you've already pulled both — Ollama just loads the one you ask for.

---

Once steps 1–8 pass, the rest of the team's code (agent loop, tool calls, API server) will work against your local Gemma instance exactly as it does on any other teammate's machine.
