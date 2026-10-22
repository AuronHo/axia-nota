"""Run after editing the game's prompts: python check_prompts.py path/to/index.html
Needs Node.js. Prints OK when prompts.py and the game build identical prompts."""
import json, subprocess, sys
from pathlib import Path
import prompts as P

cur = {"request": "Beli 2 rudal pastel, nota ke kantor.", "items": [{"name": "Rudal pastel", "qty": 2}], "to": "PT. Nota Jaya Abadi"}
nota = {"to": "PT. Nota Jaya Abadi", "date": "20 Oktober 2026", "rows": [{"name": "Rudal pastel", "qty": 2, "price": 1250000}]}
cases = [
    {"kind": "request", "lang": "id", "day": 1, "recent": []},
    {"kind": "request", "lang": "en", "day": 3, "recent": ["a", "b", "c", "d", "e", "f", "g"]},
    {"kind": "judge", "lang": "id", "day": 1, "cur": cur, "nota": nota},
    {"kind": "judge", "lang": "en", "day": 1, "cur": {**cur, "items": []}, "nota": nota},
    {"kind": "ask", "lang": "id", "day": 1, "cur": cur, "question": "Harganya berapa Bu?"},
    {"kind": "ask", "lang": "en", "day": 1, "cur": None, "question": "What now?"},
]
js = subprocess.run(["node", str(Path(__file__).with_name("check_prompts.js")), sys.argv[1]],
                    input=json.dumps(cases), capture_output=True, text=True, check=True)
want = json.loads(js.stdout)
for c, w in zip(cases, want):
    if c["kind"] == "request":
        got = P.request_prompt(c["lang"], c["day"], c["recent"])
    elif c["kind"] == "judge":
        got = P.judge_prompt(c["lang"], c["cur"], c["nota"])
    else:
        got = P.ask_prompt(c["lang"], c["cur"], c["question"])
    if got != w:
        import difflib
        print(f"MISMATCH in {c['kind']} ({c['lang']}):")
        print("\n".join(difflib.unified_diff(w.splitlines(), got.splitlines(), "game", "prompts.py", lineterm="")))
        sys.exit(1)
print("OK: prompts.py matches the game")
