"""Public front door for your model. Players' browsers talk to this, never to Ollama directly.

What it does:
  - only allows your game's website (CORS) to call it
  - forces the model name, caps the answer length, refuses huge prompts
  - rate-limits each player (default 12 calls per minute)
  - forwards to Ollama (or any OpenAI-compatible server) on your PC

Run:   uvicorn proxy:app --host 127.0.0.1 --port 8010     (from the serve folder)
Then:  cloudflared tunnel --url http://localhost:8010      (gives you a public https URL)

Settings (environment variables):
  UPSTREAM       default http://127.0.0.1:11434/v1/chat/completions
  MODEL          default bu-melati
  ALLOW_ORIGINS  default https://auronho.github.io,http://localhost:8080,http://127.0.0.1:8080,null
  RATE_PER_MIN   default 12
"""
import os, time
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:11434/v1/chat/completions")
MODEL = os.environ.get("MODEL", "bu-melati")
ORIGINS = [o.strip() for o in os.environ.get("ALLOW_ORIGINS", "https://auronho.github.io,http://localhost:8080,http://127.0.0.1:8080,null").split(",") if o.strip()]
RATE = int(os.environ.get("RATE_PER_MIN", "12"))
MAX_TOKENS = 400
MAX_PROMPT_CHARS = 9000

app = FastAPI(title="Bu Melati model proxy")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["Content-Type"], max_age=3600)
hits: dict[str, deque] = defaultdict(deque)
client = httpx.AsyncClient(timeout=120)


def client_ip(req: Request) -> str:
    # Cloudflare puts the real visitor IP here; fall back to the socket address.
    return req.headers.get("cf-connecting-ip") or (req.client.host if req.client else "unknown")


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    ip, now = client_ip(req), time.time()
    q = hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE:
        raise HTTPException(429, "Too many requests. Wait a minute.")
    q.append(now)

    body = await req.json()
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not 1 <= len(msgs) <= 4:
        raise HTTPException(400, "messages must be a list of 1-4 items")
    if sum(len(str(m.get("content", ""))) for m in msgs if isinstance(m, dict)) > MAX_PROMPT_CHARS:
        raise HTTPException(400, "prompt too long")
    clean = {
        "model": MODEL,
        "messages": [{"role": m.get("role"), "content": str(m.get("content", ""))} for m in msgs if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant")],
        "temperature": min(max(float(body.get("temperature", 0.9)), 0.0), 1.3),
        "max_tokens": min(int(body.get("max_tokens", MAX_TOKENS) or MAX_TOKENS), MAX_TOKENS),
        "stream": False,
    }
    if isinstance(body.get("response_format"), dict) and body["response_format"].get("type") == "json_object":
        clean["response_format"] = {"type": "json_object"}
    try:
        r = await client.post(UPSTREAM, json=clean)
    except httpx.HTTPError:
        raise HTTPException(502, "Model server is not running")
    if r.status_code >= 400:
        raise HTTPException(r.status_code, r.text[:300])
    return r.json()
