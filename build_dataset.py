"""Turn gold examples + generated examples into data/train.jsonl and data/val.jsonl
(chat format: {"messages": [system, user, assistant]}).

  python build_dataset.py            # gold + data/generated.jsonl
  python build_dataset.py --gold-x 3 # repeat hand-written gold examples 3x (they're the best ones)
"""
import argparse, json, random
from collections import Counter
from pathlib import Path
import prompts as P

HERE = Path(__file__).parent


def load(p):
    p = Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def gold_examples():
    req, jdg, ask = load(HERE / "gold/requests.jsonl"), load(HERE / "gold/judges.jsonl"), load(HERE / "gold/asks.jsonl")
    rows = []
    for g in req:
        others = [o["out"]["request"] for o in req if o["lang"] == g["lang"] and o is not g]
        recent = random.sample(others, random.randint(0, min(6, len(others))))
        rows.append(("request", g["lang"], P.SYSTEM_JSON, P.request_prompt(g["lang"], random.randint(1, 5), recent), json.dumps(g["out"], ensure_ascii=False)))
    for g in jdg:
        rows.append(("judge", g["lang"], P.SYSTEM_JSON, P.judge_prompt(g["lang"], g["cur"], g["nota"]), json.dumps(g["out"], ensure_ascii=False)))
    for g in ask:
        rows.append(("ask", g["lang"], P.SYSTEM_TEXT, P.ask_prompt(g["lang"], g["cur"], g["question"]), g["out"]))
    return [{"kind": k, "lang": l, "messages": [{"role": "system", "content": s}, {"role": "user", "content": u}, {"role": "assistant", "content": a}]}
            for k, l, s, u, a in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default=str(HERE / "data/generated.jsonl"))
    ap.add_argument("--gold-x", type=int, default=2)
    ap.add_argument("--val", type=float, default=.05)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    gen = load(a.generated)
    gold = [g for _ in range(a.gold_x) for g in gold_examples()]
    random.shuffle(gen)
    n_val = max(8, int(len(gen) * a.val)) if len(gen) > 40 else 0
    val, train = gen[:n_val], gen[n_val:] + gold
    if not val:  # no generated data yet: hold out a few gold rows so eval still runs
        g1 = gold_examples(); random.shuffle(g1); val = g1[:8]
    random.shuffle(train)
    (HERE / "data").mkdir(exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        with open(HERE / f"data/{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
        c = Counter((r["kind"], r["lang"]) for r in rows)
        print(f"{name}: {len(rows)} examples  " + "  ".join(f"{k}/{l}={n}" for (k, l), n in sorted(c.items())))
    if len(gen) < 300:
        print("Tip: generated set is small. Run generate_data.py for ~600+ requests for a model that doesn't repeat itself.")


if __name__ == "__main__":
    main()
