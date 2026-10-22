"""Grow the training set with a bigger "teacher" model.

The teacher sees the exact prompt the game sends, plus a few gold examples for style.
We save the plain game prompt (without the examples) + the teacher's answer, so the small
model learns to answer the real game prompts on its own.

Teachers that work (anything OpenAI-compatible):
  Ollama on your PC   --base-url http://localhost:11434/v1 --model qwen2.5:14b
  Groq free tier      --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile  (set TEACHER_API_KEY)
  OpenRouter free     --base-url https://openrouter.ai/api/v1 --model <any :free model>          (set TEACHER_API_KEY)

Example:
  python generate_data.py --n 600 --base-url http://localhost:11434/v1 --model qwen2.5:14b
Safe to stop with Ctrl+C and run again: it appends and skips duplicates.
"""
import argparse, difflib, json, os, random, sys
from pathlib import Path
import prompts as P
from teacher import chat, parse_loose

HERE = Path(__file__).parent
GOLD = HERE / "gold"

QUESTIONS = {
    "id": ["Harganya kira-kira berapa, Bu?", "Ini sebenarnya buat apa, Bu?", "Notanya ke siapa tadi, Bu?", "Berapa banyak tadi, Bu?",
           "Boleh diganti barang lain nggak?", "Bu, ini serius?", "Belinya di mana, Bu?", "Kenapa harus saya, Bu?",
           "Bayarnya pakai apa, Bu?", "Kalau barangnya nggak ada gimana?", "Ukurannya yang kayak gimana, Bu?", "Ini mendesak, Bu?"],
    "en": ["Roughly how much should I write?", "What is this actually for?", "Who was the nota for again?", "How many was it again?",
           "Can I swap it for something else?", "Ma'am, are you serious?", "Where do I even buy that?", "Why is this my job?",
           "How are we paying for this?", "What if they're sold out?", "What size do you want?", "Is this urgent?"],
}
MONTHS = {"id": ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober", "November", "Desember"],
          "en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
NONSENSE = {"id": ["Kecap manis", "Sendal jepit kiri", "Kalender 1998", "Payung bolong"],
            "en": ["Sweet soy sauce", "Left flip-flop", "1998 calendar", "Broken umbrella"]}
WRONG_TO = {"id": ["Bu Melati", "Pak Satpam", "Diri saya sendiri", "Kantin"], "en": ["Bu Melati", "The security guy", "Myself", "The canteen"]}


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def rand_price():
    return int(round(10 ** random.uniform(3, 9), -3)) or 1000


def rand_date(lang):
    return f"{random.randint(1, 28)} {random.choice(MONTHS[lang])} 2026"


def make_nota(lang, cur):
    """Build a nota for the request and decide the verdict we want the teacher to write.
    Fair notas are approved ~80% of the time; broken ones rejected ~85%; the rest are Bu Melati's whims."""
    items = cur["items"]
    rows = [{"name": i["name"] if random.random() < .6 else i["name"].split(" (")[0].lower(), "qty": i["qty"], "price": rand_price()} for i in items]
    to, problem = cur["to"], None
    kind = random.random()
    if kind < .55:
        pass
    elif kind < .68 and len(rows) > 1:
        gone = rows.pop(random.randrange(len(rows))); problem = f"the item '{gone['name']}' is missing"
    elif kind < .80:
        r = random.choice(rows); old = r["qty"]; r["qty"] = max(1, old // 10) if old >= 10 else old + random.randint(5, 30)
        problem = f"the quantity of '{r['name']}' is {r['qty']} instead of {old}"
    elif kind < .90:
        to = random.choice([w for w in WRONG_TO[lang] if w != cur["to"]]); problem = f"it is addressed to '{to}' instead of '{cur['to']}'"
    else:
        r = random.choice(rows); bad = random.choice(NONSENSE[lang]); problem = f"'{r['name']}' was replaced with '{bad}'"; r["name"] = bad
    whim = random.random() < (.2 if problem is None else .15)
    verdict = ("rejected" if whim else "approved") if problem is None else ("approved" if whim else "rejected")
    if problem is None:
        why = "the nota is correct, but she rejects it for a ridiculous personal reason" if whim else "the nota correctly covers what she asked"
    else:
        why = f"{problem}, but she approves anyway for an absurd reason (she likes the handwriting, the vibe, a number)" if whim else f"{problem}; she should point that out in character"
    nota = {"to": to, "date": rand_date(lang), "rows": rows}
    return nota, verdict, why


def examples_block(gold, lang, kind, k=3):
    pool = [g for g in gold if g["lang"] == lang]
    pick = random.sample(pool, min(k, len(pool)))
    if kind == "ask":
        lines = [f"Q: {g['question']}\nA: {g['out']}" for g in pick]
    else:
        lines = [json.dumps(g["out"], ensure_ascii=False) for g in pick]
    return "\n\nExamples of the quality and style wanted (invent something new, never copy these):\n" + "\n".join(lines)


def similar(a, pool, thr=.75):
    a = a.lower()
    return any(difflib.SequenceMatcher(None, a, b.lower()).ratio() > thr for b in pool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="how many new requests to generate (each also yields a judge, some an ask)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY", ""))
    ap.add_argument("--ask-rate", type=float, default=.35)
    ap.add_argument("--out", default=str(HERE / "data" / "generated.jsonl"))
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)

    g_req, g_jdg, g_ask = load_jsonl(GOLD / "requests.jsonl"), load_jsonl(GOLD / "judges.jsonl"), load_jsonl(GOLD / "asks.jsonl")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    seen = {"id": [g["out"]["request"] for g in g_req if g["lang"] == "id"], "en": [g["out"]["request"] for g in g_req if g["lang"] == "en"]}
    for row in load_jsonl(out):
        if row["kind"] == "request":
            seen[row["lang"]].append(json.loads(row["messages"][2]["content"])["request"])
    made = {"request": 0, "judge": 0, "ask": 0}; fails = 0

    def call(user, json_mode):
        system = P.SYSTEM_JSON if json_mode else P.SYSTEM_TEXT
        text, _ = chat(a.base_url, a.model, system, user, json_mode=json_mode, api_key=a.api_key)
        return text

    def save(kind, lang, system, user, answer):
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "lang": lang, "messages": [
                {"role": "system", "content": system}, {"role": "user", "content": user}, {"role": "assistant", "content": answer}]}, ensure_ascii=False) + "\n")
        made[kind] += 1

    try:
        while made["request"] < a.n:
            lang = random.choice(["id", "en"])
            day = random.choice([1, 1, 2, 2, 3, 4, 5, 6])
            recent = random.sample(seen[lang], min(len(seen[lang]), random.randint(0, 6)))
            game_prompt = P.request_prompt(lang, day, recent)
            try:
                obj = parse_loose(call(game_prompt + examples_block(g_req, lang, "request"), True))
                err = P.check_request(obj)
                if err:
                    raise RuntimeError(err)
                if similar(obj["request"], seen[lang]):
                    raise RuntimeError("too similar to an earlier request")
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"  skip request: {e}"); continue
            save("request", lang, P.SYSTEM_JSON, game_prompt, json.dumps(obj, ensure_ascii=False))
            seen[lang].append(obj["request"])

            nota, verdict, why = make_nota(lang, obj)
            jp = P.judge_prompt(lang, obj, nota)
            note = f'\n\nWriter\'s note (hidden from the player): the verdict must be "{verdict}" because {why}. Pick patience and mood to match.'
            try:
                j = parse_loose(call(jp + note + examples_block(g_jdg, lang, "judge"), True))
                err = P.check_judge(j) or (None if j["verdict"] == verdict else "wrong verdict")
                if err:
                    raise RuntimeError(err)
                save("judge", lang, P.SYSTEM_JSON, jp, json.dumps(j, ensure_ascii=False))
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"  skip judge: {e}")

            if random.random() < a.ask_rate:
                cur = obj if random.random() > .08 else None
                q = random.choice(QUESTIONS[lang])
                ap_ = P.ask_prompt(lang, cur, q)
                try:
                    ans = call(ap_ + examples_block(g_ask, lang, "ask"), False).strip().strip('"')
                    err = P.check_ask(ans)
                    if err:
                        raise RuntimeError(err)
                    save("ask", lang, P.SYSTEM_TEXT, ap_, ans)
                except Exception as e:  # noqa: BLE001
                    fails += 1; print(f"  skip ask: {e}")
            print(f"[{made['request']}/{a.n}] requests={made['request']} judges={made['judge']} asks={made['ask']} skipped={fails}  | {obj['request'][:70]}")
    except KeyboardInterrupt:
        print("\nStopped. Run again to continue.")
    print(f"Saved to {out}: {made}")


if __name__ == "__main__":
    sys.exit(main())
