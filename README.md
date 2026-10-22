# Axia Nota

A comedy 3D browser game set in a tiny office in Batam. You are **Axia**, the company's only nota
clerk. Your boss, **Bu Melati**, is played by an AI, and her requests make no sense: buy two pastel
missiles, a cat-shaped cloud over Nagoya Hill, forty ducks for the night shift. Your job is to write
a proper nota for every one of them, and survive the week.

**▶ Play it: [auronho.github.io/axia-nota](https://auronho.github.io/axia-nota/)**
Runs in any modern browser, desktop or phone. Nothing to install.

## How it plays

- **A Monday–Friday work week.** Each day has 5 notas, and at least 3 must be approved or you're
  fired. An office clock runs from 08:00 to 17:00 as you work.
- **Fill in a real carbon-copy nota:** who it's for, quantities, items and prices (it adds up the
  totals), then sign it and hand it over. Bu Melati judges it, usually fairly, and stamps it.
- **Ask her to clarify** (costs Axia some patience), or press **Gamau** to refuse. Three refusals
  and Axia resigns.
- **Four endings:** a resignation letter, a termination letter, a payslip paid in crackers, or a
  promotion to Head of the Nota Division. The letters follow real Indonesian business-letter
  format, down to the meterai.
- **Cutscenes:** Axia walks in on Monday morning, falls asleep at his desk each night, and gets
  yelled awake at sunrise.
- Indonesian and English. Progress saves in the player's browser.

## How the AI works

Bu Melati is **Qwen2.5-1.5B-Instruct, fine-tuned with LoRA** to play this one character and always
answer in the exact JSON the game needs.

```
Browser (this page)  →  proxy (CORS + rate limit)  →  Ollama  →  bu-melati model
GitHub Pages            FastAPI + Cloudflare tunnel      GGUF, 8-bit, ~1.6 GB, on an RTX 5060
```

- **Training data:** 75 hand-written examples set the voice, and a larger "teacher" model writes
  hundreds more in the same style. Fair and broken notas are generated on purpose, so the judge
  learns when to approve and when to reject.
- **Training:** LoRA (rank 16) with TRL's SFTTrainer, trained only on the boss's answers, on a single
  8 GB GPU. The result is merged, converted to GGUF with llama.cpp and served by Ollama.
- **Evaluation:** `eval.py` plays rounds against any model and scores game-ready JSON, verdict
  accuracy on clear-cut notas, repetition and speed, so the fine-tune can be compared with the base
  model.
- **Serving safely:** the browser never talks to the model directly. A small FastAPI proxy only
  accepts this site, forces the model name, caps answer length and rate-limits each player.
- **Always playable:** if the model server is offline, the game falls back to built-in lines. Inside
  claude.ai it can use Claude instead.

## Tech

| Part | Built with |
|---|---|
| Game | One HTML file, vanilla JavaScript, Three.js (toon shading, cutscene camera), no build step |
| Model | Qwen2.5-1.5B-Instruct, PyTorch, Transformers, TRL, PEFT (LoRA), llama.cpp GGUF |
| Serving | Ollama, FastAPI, Uvicorn, Cloudflare Tunnel (or a free Hugging Face Space) |

## Repo layout

| Path | What it is |
|---|---|
| `index.html` | The whole game |
| `SETUP.md` | Step-by-step: train and serve the model on Windows, then publish |
| `TRAINING.md` | What each step does and why, plus how a web page talks to a model |
| `prompts.py`, `check_prompts.py` | The game's prompts in Python, and a check that they still match the game |
| `gold/` | Hand-written example dialogue |
| `generate_data.py`, `build_dataset.py` | Make the training set with a teacher model |
| `train.py`, `eval.py` | Fine-tune, then score the result |
| `serve/` | Ollama Modelfile, the proxy, and a one-click start script |
| `space/` | Files for free hosting on a Hugging Face Space |

## Run it yourself

- **Just play:** open `index.html` in a browser. Without a model server it uses the built-in lines.
- **With your own model:** follow [SETUP.md](SETUP.md).

---

Made by **Auron Rafael Hoverio** · [GitHub](https://github.com/AuronHo) · [LinkedIn](https://www.linkedin.com/in/auron-rafael-hoverio)
