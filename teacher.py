"""Tiny client for any OpenAI-compatible chat endpoint (Ollama, LM Studio, llama.cpp server,
Groq, OpenRouter, your own proxy). Used by generate_data.py and eval.py."""
import json, re, time
import requests


def chat(base_url: str, model: str, system: str, user: str, *, json_mode: bool, api_key: str = "",
         temperature: float = 0.9, max_tokens: int = 500, timeout: int = 180) -> tuple[str, float]:
    """Returns (reply_text, seconds). Raises RuntimeError with a readable message on failure."""
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if re.search(r"qwen3", model, re.I):
        user += "\n/no_think"
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    t0 = time.time()
    r = requests.post(url, headers=headers, json={**body, "response_format": {"type": "json_object"}} if json_mode else body, timeout=timeout)
    if json_mode and r.status_code in (400, 422):  # some servers reject response_format; retry plain
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code == 429:
        raise RuntimeError("rate limited (429)")
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    d = r.json()
    text = (d.get("choices") or [{}])[0].get("message", {}).get("content") or d.get("message", {}).get("content") or ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    if not text:
        raise RuntimeError("empty reply")
    return text, time.time() - t0


def parse_loose(text: str):
    """Same tolerant JSON parsing as the game: whole text, a ``` fence, or first { .. last }."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            pass
    return None
