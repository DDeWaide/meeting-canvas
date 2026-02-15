# Meeting Canvas

Real-time AI meeting visualization. Speak into your mic — AI agents listen, extract, and organize your conversation into a live visual whiteboard with zones, hierarchy, connections, and notes.

![Meeting Canvas](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/license-MIT-green)

## What It Does

- **Live transcription** via Deepgram Nova-3 (speaker diarization)
- **Multi-agent extraction** — 3-6 AI agents dynamically scale based on conversation complexity
- **Visual whiteboard** — auto-organized into zones, groups, steps, and connections
- **Executive summary** — one-click Sonnet-powered meeting summary
- **Export** — download as `.drawio` file (works in Lucidchart, Draw.io, etc.)

## Quick Start (2 minutes)

### 1. Get API Keys

| Service | Free Tier | Link |
|---------|-----------|------|
| **Deepgram** | $200 free credit | [console.deepgram.com](https://console.deepgram.com) |
| **Anthropic** | Pay-as-you-go (~$0.01/min) | [console.anthropic.com](https://console.anthropic.com) |

### 2. Install & Run

```bash
# Option A: Set env vars
# Mac/Linux:
export DEEPGRAM_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# Windows PowerShell:
$env:DEEPGRAM_API_KEY="your-key"
$env:ANTHROPIC_API_KEY="your-key"

python -m uvicorn app:app --reload

# Option B: Just run it (enter keys in browser) ← easiest
python -m uvicorn app:app --reload
```

### 3. Open

Go to **http://localhost:8000** → enter API keys if prompted → click **Start Recording** → talk.

## Hosted Deployment

If you want to host this for others (they bring their own API keys):

### Docker

```bash
docker build -t meeting-canvas .
docker run -p 8000:8000 meeting-canvas
```

### Railway / Render / Fly.io

1. Push `app.py`, `frontend.html`, `requirements.txt`, and `Dockerfile` to a repo
2. Connect to your preferred platform
3. Deploy — no env vars needed (users enter their own keys)

### HTTPS Required

Browsers require HTTPS for microphone access on non-localhost domains. Make sure your deployment has SSL (most platforms handle this automatically).

## How It Works

```
Mic Audio → Deepgram (transcription) → Agent Pool → Claude Haiku (extraction) → Visual Canvas
                                           ↕
                                    Dynamic scaling (3-6 agents)
                                    Content classification
                                    Specialization shifting
```

**Capacitor Architecture:** Agents have specializations (structural, notes, relationships, ambiguity) and dynamically scale up when conversation gets complex, scale down during quiet periods. A merge layer deduplicates across agents.

## Cost

Roughly **$0.01-0.02 per minute** of meeting (Haiku extraction). Summary generation uses Sonnet (~$0.02 per summary). Deepgram has a generous free tier.

## Files

```
app.py           — FastAPI backend, agent pool, Deepgram/Anthropic integration
frontend.html    — Single-file frontend (no build step)
requirements.txt — Python dependencies
Dockerfile       — Container deployment
```

## License

MIT — use it, modify it, ship it.
