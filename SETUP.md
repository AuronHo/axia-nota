# Bu Melati model: step-by-step setup (Windows, RTX 5060 8 GB)

Everything lives inside `C:\axia-nota-model`, **including Python itself**. Nothing gets installed
system-wide except the Ollama app (already installed). Deleting the folder removes almost everything.

| Folder | What's in it |
|---|---|
| `tools\uv.exe` | A small tool that downloads Python and installs packages |
| `.python\` | Python 3.11, downloaded by uv just for this project |
| `.venv\` | The virtual environment with all the Python packages |
| `.uv-cache\`, `.pip-cache\` | Package download caches |
| `.hf-cache\` | The Qwen base model downloaded for training |
| `.ollama-models\` | All Ollama models, including your `bu-melati` |
| `serve\cloudflared.exe` | The tunnel that gives you a public address |

**Use Command Prompt, not PowerShell.** In Windows Terminal, click the **▾** next to the tab and
choose **Command Prompt**. Several commands below behave differently in PowerShell.

---

## Phase 1: Get the kit and tools (~5 min)

**1.** Unzip `axia-nota-model.zip` so the files sit directly in `C:\axia-nota-model`. You should
have `C:\axia-nota-model\SETUP.md`, not `C:\axia-nota-model\axia-nota-model\SETUP.md`.

**2.** Open Command Prompt and download uv:
```bat
cd C:\axia-nota-model
mkdir tools
curl -L -o tools\uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
tar -xf tools\uv.zip -C tools
tools\uv.exe --version
```
✅ The last line prints something like `uv 0.x.y`.

**3.** Download cloudflared:
```bat
curl -L -o serve\cloudflared.exe https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
serve\cloudflared.exe --version
```
✅ It prints a version number.

## Phase 2: Python and the virtual environment, inside the folder (~10 min)

**4.** Download Python 3.11 into the project and create the venv:
```bat
set UV_PYTHON_INSTALL_DIR=C:\axia-nota-model\.python
set UV_CACHE_DIR=C:\axia-nota-model\.uv-cache
tools\uv.exe venv --python 3.11 --seed .venv
```
✅ You see `Using CPython 3.11.x` and `Creating virtual environment at: .venv`.

**5.** Make the venv keep every cache inside the project, then switch it on:
```bat
echo set HF_HOME=C:\axia-nota-model\.hf-cache>> .venv\Scripts\activate.bat
echo set UV_CACHE_DIR=C:\axia-nota-model\.uv-cache>> .venv\Scripts\activate.bat
echo set PIP_CACHE_DIR=C:\axia-nota-model\.pip-cache>> .venv\Scripts\activate.bat
.venv\Scripts\activate
python --version
where python
```
✅ The prompt starts with `(.venv)`, the version is `Python 3.11.x`, and the first path is
`C:\axia-nota-model\.venv\Scripts\python.exe`.

**6.** Install the packages. Your RTX 50-series card needs the `cu128` PyTorch build, and torch is
a big download (~3 GB):
```bat
tools\uv.exe pip install torch --index-url https://download.pytorch.org/whl/cu128
tools\uv.exe pip install -r requirements.txt
```

**7.** Check that the GPU works from Python:
```bat
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.zeros(1).cuda())"
```
✅ It prints `NVIDIA GeForce RTX 5060` and `tensor([0.], device='cuda:0')`.
If it says `no kernel image is available`, the wrong torch build got in. Rerun step 6.

> **Every new terminal from now on starts with:**
> ```bat
> cd C:\axia-nota-model
> .venv\Scripts\activate
> ```

## Phase 3: Ollama models inside the folder (~10 min)

**8.** Point Ollama's storage at the project:
```bat
setx OLLAMA_MODELS "C:\axia-nota-model\.ollama-models"
```
Right-click the Ollama icon in the system tray, choose **Quit**, then start Ollama again from the
Start menu. Open a **new** terminal and activate the venv.

**9.** Download the teacher model and the base model you'll compare against:
```bat
ollama pull qwen2.5:7b
ollama pull qwen2.5:1.5b
dir .ollama-models
```
✅ `.ollama-models` now contains `blobs` and `manifests`.

## Phase 4: Make the training data (1–3 hours, mostly waiting)

**10.** Do a test run with 5 requests:
```bat
python generate_data.py --n 5 --model qwen2.5:7b
```
✅ It ends with `Saved to ... {'request': 5, 'judge': 5, ...}`. A few `skip` lines are normal.

**11.** Open `data\generated.jsonl` in VS Code and read the `assistant` parts. If Bu Melati sounds
right, run the full set:
```bat
python generate_data.py --n 600 --model qwen2.5:7b
```
Ctrl+C stops it safely, and running it again continues where it stopped.

*If 7B's Indonesian jokes feel flat,* use a stronger free teacher. Make a free key at
console.groq.com, then run:
```bat
set TEACHER_API_KEY=your_key
python generate_data.py --n 600 --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile
```

**12.** Skim the file and delete any bad lines, then build the dataset:
```bat
python build_dataset.py
```
✅ It prints `train: ... examples` and `val: ... examples`.

## Phase 5: Train (~20–45 min)

**13.** Close Chrome, Edge, Teams and Steam to free GPU memory, then run:
```bat
python train.py
```
The first run downloads Qwen2.5-1.5B (~3 GB) into `.hf-cache`. Watch `eval_loss` after each epoch:
it should go down.
✅ It ends with `Merged model saved to ...\out\merged`.
If you get `CUDA out of memory`, run `python train.py --batch 1 --grad-accum 16` instead.

## Phase 6: Test your model (~10 min)

**14.** Convert the model to GGUF, load it into Ollama, and score it against the base model.

First get llama.cpp's converter. It's a one-time download and doesn't need building:
```bat
git clone --depth 1 https://github.com/ggml-org/llama.cpp tools\llama.cpp
tools\uv.exe pip install sentencepiece protobuf pyyaml tqdm
```
Don't install llama.cpp's own `requirements.txt`: it would replace your CUDA PyTorch with a
CPU-only one.

Convert, build, and test:
```bat
python tools\llama.cpp\convert_hf_to_gguf.py out\merged --outfile out\bu-melati-q8_0.gguf --outtype q8_0
ollama create bu-melati -f serve\Modelfile
python eval.py --model qwen2.5:1.5b
python eval.py --model bu-melati
```
✅ `out\bu-melati-q8_0.gguf` is about 1.6 GB, `ollama create` ends with `success`, and `bu-melati`
shows a higher `game-ready %` and better verdict agreement than `qwen2.5:1.5b`.

Why GGUF: newer Ollama versions can't import Qwen2 safetensors directly ("unsupported MLX
architecture"). GGUF is the format Ollama and llama.cpp run natively, and the same file is what
you'd upload for the Hugging Face Space option.

**15.** *(Optional)* Play it locally before going public:
1. Run `setx OLLAMA_ORIGINS "*"`, then quit and restart Ollama.
2. Double-click `index.html`.
3. Open ⚙ and choose **Server / local model**. Set the endpoint to
   `http://localhost:11434/v1/chat/completions` and the model to `bu-melati`.
4. Click **Tes koneksi**, then play.

When you're done, remove that permission again, because it lets any website you visit use your
local Ollama:
```bat
reg delete HKCU\Environment /v OLLAMA_ORIGINS /f
```

## Phase 7: Make it public (~10 min)

**16.** Open a new terminal, activate the venv, then start the server:
```bat
cd serve
start-server.bat
```
A second window opens for the proxy. The first window shows a line with
`https://something.trycloudflare.com`. Copy that address.

**17.** Open that address with `/health` added in Chrome.
✅ It shows `{"ok":true,"model":"bu-melati"}`.

**18.** Open `index.html` in VS Code. Near the top of the script, paste your address:
```js
const GAME_CONFIG = { serverUrl: "https://something.trycloudflare.com/v1/chat/completions", serverModel: "bu-melati" };
```

## Phase 8: Put the game online (~2 min)

The game lives in its own repo, github.com/AuronHo/axia-nota, and GitHub Pages serves `index.html`
from it. If this folder isn't linked to that repo yet, do "Link this folder to GitHub" below once.

**19.** After changing `GAME_CONFIG`, push it:
```bat
git add index.html
git commit -m "Update model server address"
git push
```

**20.** After about a minute, open `https://auronho.github.io/axia-nota/`.
✅ Bu Melati answers from your model. Test it from your phone too.

### Link this folder to GitHub (once)

This keeps everything you already have (venv, models, training data) and connects the folder to
the repo. The big and private folders are listed in `.gitignore`, so they never get uploaded.
```bat
cd C:\axia-nota-model
git init -b main
git remote add origin https://github.com/AuronHo/axia-nota.git
git fetch origin
git reset origin/main
git checkout -- .
git status
```
`git checkout -- .` replaces the kit files with the repo's latest versions. If `git status` still
lists a `game\` folder, delete it: the game is now `index.html` in the main folder.

---

## Day-to-day

- **To let people play:** keep Ollama running and `serve\start-server.bat` open.
- **If the PC is off:** the game falls back to its built-in script.
- **After restarting `start-server.bat`:** you get a new address. Redo steps 18–19 with it.

## Cleanup (removes everything)

```bat
rmdir /s /q C:\axia-nota-model
reg delete HKCU\Environment /v OLLAMA_MODELS /f
```
Then restart Ollama. Ollama itself stays installed; uninstall it from Windows Settings if you want.
