# Bu Melati model: train your own boss for Axia Nota

This kit fine-tunes **Qwen2.5-1.5B-Instruct** on your own GPU so it plays Bu Melati, then puts it
behind a public URL. Players open the game link and play straight away. The model runs on your
server, not in their browser, so they download nothing.

```
 game page (GitHub Pages)  ──fetch──▶  proxy (CORS + rate limit)  ──▶  Ollama / llama.cpp  ──▶  bu-melati model
   auronho.github.io/axia-nota          your PC + Cloudflare tunnel        on your PC
                                        or a free Hugging Face Space (no proxy needed)
```

## What's in here

| File | What it does |
|---|---|
| `prompts.py` | The exact prompts the game sends, plus checks that an answer is game-ready |
| `check_prompts.py` | Confirms `prompts.py` still matches the game after you edit the game |
| `gold/*.jsonl` | 75 hand-written examples of Bu Melati at her best (requests, verdicts, answers) |
| `generate_data.py` | Uses a bigger "teacher" model to write hundreds more examples in the same style |
| `build_dataset.py` | Mixes gold + generated into `data/train.jsonl` and `data/val.jsonl` |
| `train.py` | LoRA fine-tune on your GPU, then merges into a full model in `out/merged` |
| `eval.py` | Plays rounds against any model and scores it the way the game uses it |
| `serve/` | Ollama `Modelfile`, the public `proxy.py`, and a Windows start script |
| `space/` | Files for free hosting on a Hugging Face Space |

## 0. Setup (Windows, once)

Follow **SETUP.md**. It's the exact click-by-click path for this PC (RTX 5060, 8 GB), with Python,
the venv, every cache and the Ollama models kept inside this folder. The sections below explain
what each step does and why.

## 1. Make the training data

The 75 gold examples teach the style. A teacher model writes a few hundred more so your model
doesn't repeat itself. Any OpenAI-compatible model works. A bigger teacher gives a funnier student.

**Teacher on your own GPU (free, offline):**
```bat
ollama pull qwen2.5:7b
python generate_data.py --n 600 --base-url http://localhost:11434/v1 --model qwen2.5:7b
```
On a 12 GB+ card, `qwen2.5:14b` writes better jokes. On 8 GB it spills into system RAM and crawls, so 7B is the default here. 7B's Indonesian jokes are weaker, so check the output, or use the free API teacher below.

**Teacher from a free API (stronger, needs a key):**
```bat
set TEACHER_API_KEY=your_groq_key
python generate_data.py --n 600 --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile
```

Each request also produces a judged nota. About a third also produce a question-and-answer. The
script builds both fair and broken notas (missing item, wrong quantity, wrong recipient), then
tells the teacher which verdict to give. That way the labels stay correct, and Bu Melati's random
whims stay at about 1 in 5. You can stop it with Ctrl+C and run it again: it appends and skips
near-duplicates.

**Read 20 lines of `data/generated.jsonl` before training.** Your model will copy whatever is in
there. Delete any lines that are bad.

```bat
python build_dataset.py
```

## 2. Train

```bat
python train.py
```
Expect about 15–40 minutes for ~1,500 examples on a mid-range RTX card. Watch `eval_loss` after
each epoch: it should drop and then flatten. If it rises in epoch 3, run again with `--epochs 2`.
Output:
- `out/lora` is the small adapter, about 70 MB.
- `out/merged` is the full fine-tuned model, which Ollama and llama.cpp load.

## 3. Run it with Ollama and compare

```bat
git clone --depth 1 https://github.com/ggml-org/llama.cpp tools\llama.cpp
python tools\llama.cpp\convert_hf_to_gguf.py out\merged --outfile out\bu-melati-q8_0.gguf --outtype q8_0
ollama create bu-melati -f serve/Modelfile
ollama pull qwen2.5:1.5b
python eval.py --model qwen2.5:1.5b
python eval.py --model bu-melati
```
`eval.py` plays 20 rounds. It reports how often answers are game-ready, how often the verdict
matches on clear-cut notas, how many requests repeat, and the speed. The fine-tune should beat the
base model on game-ready answers and on verdicts.

To play locally right away: start Ollama with browser access allowed, open the game file, then go
to ⚙ → *Server / local model* and set model `bu-melati`.

```bat
setx OLLAMA_ORIGINS "*"
```
After running that, quit Ollama from the tray and start it again.

## 4. Make it public (players need nothing)

### Option A: serve from your PC (fast, free, PC must stay on)

```bat
cd serve
start-server.bat
```
This starts `proxy.py` on port 8010 and opens a Cloudflare quick tunnel. Look for a line like
`https://random-words.trycloudflare.com`. Your public endpoint is that address plus
`/v1/chat/completions`.

The proxy matters because players never touch Ollama directly. It only accepts calls from your
game's site, forces the model to `bu-melati`, caps answer length, and limits each player to 12
calls a minute. Change these with the environment variables listed at the top of `proxy.py`.

Quick-tunnel addresses change every time you restart. For a permanent address, create a named
tunnel on a domain you own (see the Cloudflare docs), or use Option B.

### Option B: free Hugging Face Space (always on, slower)

1. Convert the merged model to GGUF:
   ```bat
   git clone https://github.com/ggml-org/llama.cpp
   pip install -r llama.cpp\requirements.txt
   python llama.cpp\convert_hf_to_gguf.py out\merged --outfile out\bu-melati-q8_0.gguf --outtype q8_0
   ```
   For the smaller Q4_K_M, download a llama.cpp release and run
   `llama-quantize out\bu-melati-q8_0.gguf out\bu-melati-Q4_K_M.gguf Q4_K_M`.
2. Upload it as a model repo:
   ```bat
   pip install huggingface_hub
   huggingface-cli login
   huggingface-cli upload AuronHo/bu-melati-1.5b-GGUF out\bu-melati-Q4_K_M.gguf
   ```
3. Create a new Space with the **Docker** SDK, then upload `space/Dockerfile` and `space/README.md`.
   If your repo name or quant differs, change the `-hf` line.
4. Your endpoint is `https://<user>-<space>.hf.space/v1/chat/completions`.

Things to know about the free CPU Space:
- Replies take noticeably longer than on your GPU, roughly 10–30 seconds each.
- It sleeps after a while with no visitors, and the first call wakes it.
- It has no rate limit of its own.

If your browser shows a CORS error from the Space, or you want rate limits, run `proxy.py` on a
second Space and point its `UPSTREAM` at the first.

## 5. Point the game at it and publish the game

Open `index.html` (the game). Near the top of the script, set your endpoint:

```js
const GAME_CONFIG = { serverUrl: "https://random-words.trycloudflare.com/v1/chat/completions", serverModel: "bu-melati" };
```

Then publish the game on GitHub Pages:
1. Put the game in your `AuronHo.github.io` repo as `axia-nota/index.html`.
2. Push it. The game is live at `https://auronho.github.io/axia-nota/`.

Players open the link and Bu Melati answers from your model. If the server is off, the game falls
back to its built-in script, so it never breaks.

The claude.ai version of the game can't call your server, because pages inside claude.ai may only
talk to Claude. Use GitHub Pages for the public version.

## How a web page talks to a model (the learning part)

- **The browser only sends a `fetch()`.** The game posts JSON to `/v1/chat/completions`: a system
  message, the game prompt, and `response_format: json_object` when it needs JSON. That is the
  OpenAI chat format, which Ollama, llama.cpp, LM Studio, Groq and OpenRouter all accept. So
  swapping brains only means changing the URL.
- **CORS.** A browser refuses to read a response from another site unless that server replies
  with `Access-Control-Allow-Origin` naming your page. That's why Ollama needs `OLLAMA_ORIGINS`,
  and why `proxy.py` lists your GitHub Pages domain.
- **HTTPS.** A page served over https may not call a plain-http server, except `localhost`. The
  Cloudflare tunnel and HF Spaces both give you https.
- **Never put a secret key in the page.** Anyone can read it with View Source. That's another
  reason the proxy sits in front of your model.
- **Why fine-tune?** A 1.5B model with only the prompt often breaks the JSON or forgets the
  character. Training on a few hundred perfect answers teaches the format and the voice, so a
  small, cheap model behaves like the big teacher on this one job.

## Changing the game's prompts later

The model learns the exact prompt text. If you edit `requestPrompt`, `judgePrompt` or `askPrompt`
in the game, mirror the change in `prompts.py`, then check them with
`python check_prompts.py index.html`. It needs Node.js and prints `OK` when they match.
After that, regenerate and retrain. Small wording changes usually still work, but big ones need
retraining.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | `python train.py --batch 1 --grad-accum 16`, or `--max-len 1024` |
| `train.py` says no GPU | Reinstall torch with the CUDA command from step 0 |
| The game says it can't reach the model (CORS) | Restart Ollama after `setx OLLAMA_ORIGINS "*"`; for the proxy, add your page's origin to `ALLOW_ORIGINS` |
| Many "skip request" lines in generate_data | The teacher is too small or ignores JSON; use a bigger teacher |
| Model repeats the same jokes | Generate more data (`--n 1200`) and train 2 epochs instead of 3 |
