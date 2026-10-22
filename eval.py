"""Score a model the way the game uses it. Run it on the base model and on your fine-tune to compare.

  python eval.py --model qwen2.5:1.5b       # base model in Ollama
  python eval.py --model bu-melati          # your fine-tune in Ollama
  python eval.py --base-url https://xxxx.trycloudflare.com/v1 --model bu-melati   # through the public proxy

Reports: valid JSON %, game-ready %, judge verdict agreement on clearly fair/broken notas, variety, speed.
"""
import argparse, difflib, json, random, statistics
from pathlib import Path
import prompts as P
from generate_data import QUESTIONS, load_jsonl, make_nota
from teacher import chat, parse_loose

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="bu-melati")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--n", type=int, default=20, help="rounds (each = 1 request + 1 judge + 1 ask)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)
    gold = load_jsonl(HERE / "gold/requests.jsonl")
    stats = {k: {"ok": 0, "json": 0, "n": 0} for k in ("request", "judge", "ask")}
    agree = agree_n = 0
    secs, made, samples = [], [], []

    for i in range(a.n):
        lang = random.choice(["id", "en"])
        recent = [g["out"]["request"] for g in random.sample(gold, 4) if g["lang"] == lang]
        # request
        stats["request"]["n"] += 1
        try:
            text, s = chat(a.base_url, a.model, P.SYSTEM_JSON, P.request_prompt(lang, random.randint(1, 5), recent), json_mode=True, api_key=a.api_key)
            secs.append(s); obj = parse_loose(text)
            if obj is not None: stats["request"]["json"] += 1
            if obj is not None and not P.check_request(obj):
                stats["request"]["ok"] += 1; made.append(obj["request"])
                if len(samples) < 3: samples.append(("request", obj))
            else:
                obj = None
        except Exception as e:  # noqa: BLE001
            print("  request error:", e); obj = None
        # judge (on a gold request if the model's own one failed)
        cur = obj or random.choice([g["out"] for g in gold if g["lang"] == lang])
        nota, verdict, why = make_nota(lang, cur)
        stats["judge"]["n"] += 1
        try:
            text, s = chat(a.base_url, a.model, P.SYSTEM_JSON, P.judge_prompt(lang, cur, nota), json_mode=True, api_key=a.api_key)
            secs.append(s); j = parse_loose(text)
            if j is not None: stats["judge"]["json"] += 1
            if j is not None and not P.check_judge(j):
                stats["judge"]["ok"] += 1
                if "absurd" not in why and "ridiculous" not in why:  # only count clear-cut cases
                    agree_n += 1; agree += j["verdict"] == verdict
                if len(samples) < 5: samples.append(("judge", j))
        except Exception as e:  # noqa: BLE001
            print("  judge error:", e)
        # ask
        stats["ask"]["n"] += 1
        try:
            text, s = chat(a.base_url, a.model, P.SYSTEM_TEXT, P.ask_prompt(lang, cur, random.choice(QUESTIONS[lang])), json_mode=False, api_key=a.api_key)
            secs.append(s); stats["ask"]["json"] += 1
            if not P.check_ask(text): stats["ask"]["ok"] += 1
        except Exception as e:  # noqa: BLE001
            print("  ask error:", e)
        print(f"round {i+1}/{a.n} done")

    dup = sum(1 for k, x in enumerate(made) if any(difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio() > .75 for y in made[:k]))
    print("\n=== " + a.model + " ===")
    for k, v in stats.items():
        n = max(1, v["n"])
        print(f"{k:8s} parsed {v['json']*100//n:3d}%   game-ready {v['ok']*100//n:3d}%")
    if agree_n:
        print(f"judge agrees with the expected verdict on clear cases: {agree*100//agree_n}% ({agree}/{agree_n})")
    print(f"near-duplicate requests: {dup}/{len(made)}")
    if secs:
        print(f"latency: median {statistics.median(secs):.1f}s, slowest {max(secs):.1f}s")
    print("\nsamples:")
    for k, o in samples:
        print(f"  [{k}] {json.dumps(o, ensure_ascii=False)[:220]}")


if __name__ == "__main__":
    main()
