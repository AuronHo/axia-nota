"""Prompt builders mirrored 1:1 from the Axia Nota game (index.html).

The model is trained on exactly the prompts the game sends, so keep this file in sync
with the game's requestPrompt / judgePrompt / askPrompt functions. `check_prompts.js`
can be used to confirm both sides still produce identical text.
"""

SYSTEM_JSON = "You write dialogue for a comedy game. Reply with one valid JSON object and nothing else."
SYSTEM_TEXT = "You write dialogue for a comedy game. Reply in plain text only."

WORLD = """You write lines for a comedy browser game called "Axia Nota", set in a small trading office (PT. Nota Jaya Abadi) in Batam, Indonesia.
Characters:
- Bu Melati: the female director and Axia's boss. Confident, dramatic and completely unhinged. Her requests are absurd but she delivers them as if they were normal business. She sometimes blurts out an illogical statement that has nothing to do with anything.
- Axia: a young office clerk whose whole job is writing "nota" (handwritten sales receipts). He is fed up, but he still does his job.
Humor rules: cartoonish, harmless slapstick absurdity (the kind of joke is "buy a missile to bomb Batam and make a nota for it"). Never give real-world how-to details, never name real living people, no slurs, nothing sexual, nothing cruel about real groups."""

LANG = {
    "id": "Indonesian. Casual spoken Indonesian like a Batam office (bahasa gaul kantor is fine, e.g. 'kamu', 'dong', 'ya'). Item names in Indonesian.",
    "en": "English, casual and punchy. Keep Indonesian flavour words like 'nota', 'Bu', 'Batam' as they are.",
}
STAMP_EXAMPLES = {"id": "LUNAS, DITOLAK, MENCURIGAKAN", "en": "PAID, REJECTED, SUSPICIOUS"}
MOODS = ["sweet", "furious", "dramatic", "dreamy", "suspicious"]


def rp(n) -> str:
    """Same as JS (n).toLocaleString('id-ID') for whole numbers: 1.250.000"""
    return f"{int(n or 0):,}".replace(",", ".")


def request_prompt(lang: str, day: int, recent: list[str]) -> str:
    rec = "\n".join("- " + r for r in recent[-6:]) or "- (none yet)"
    return f"""{WORLD}

Write Bu Melati's next request to Axia. It must be something Axia can write a nota for: name 1 to 4 concrete things to buy, each with a quantity, and say who the nota should be addressed to. Escalate the weirdness with the day number (day {day}). Vary the topic, the mood and the shape of the joke. Mention local Batam/Riau Islands places sometimes.
Do not repeat or closely copy these recent requests:
{rec}

Language for every text field: {LANG[lang]}

Reply with only this JSON object:
{{"request":"what she says to Axia, 1-3 sentences, includes items, quantities and who the nota is for","aside":"an unrelated illogical remark she adds; use an empty string about half the time","mood":"one of sweet, furious, dramatic, dreamy, suspicious","axia":"Axia's short annoyed inner thought about this request, max 12 words","items":[{{"name":"short item name","qty":1}}],"to":"who the nota is addressed to"}}"""


def judge_prompt(lang: str, cur: dict, nota: dict) -> str:
    rows = "\n".join(
        f"  {r['qty']} x {r['name']} @ Rp {rp(r['price'])} = Rp {rp(r['qty'] * r['price'])}" for r in nota["rows"]
    )
    expected = "; ".join(f"{i['qty']} x {i['name']}" for i in cur["items"]) or "(see request)"
    total = sum(r["qty"] * r["price"] for r in nota["rows"])
    return f"""{WORLD}

Bu Melati asked Axia: "{cur['request']}"
What she expected: {expected}, addressed to "{cur['to']}".

Axia handed her this nota:
  To: {nota['to']}
  Date: {nota['date']}
{rows}
  Total: Rp {rp(total)}
  Signed by Axia: yes

Judge it as Bu Melati. Be mostly fair: approve when the nota reasonably covers what she asked (spelling and exact wording don't matter, prices are Axia's guess). Reject when items are missing, quantities are clearly wrong, it is addressed to the wrong party, or it is obvious nonsense. About one time in five, follow an absurd personal whim instead (reject for a ridiculous reason, or approve a wrong nota because she likes the handwriting). She has strong, illogical opinions about prices; comment on them sometimes.

Language for every text field: {LANG[lang]}

Reply with only this JSON object:
{{"verdict":"approved" or "rejected","reply":"her reaction to Axia, 1-3 sentences, in character","stamp":"1-2 word rubber stamp text in capitals, e.g. {STAMP_EXAMPLES[lang]}","patience":integer from -25 to 5 for how this affects Axia's patience (unfair rejections hurt most, fair approvals cost little),"mood":"one of sweet, furious, dramatic, dreamy, suspicious","axia":"Axia's inner thought, max 12 words"}}"""


def ask_prompt(lang: str, cur: dict | None, question: str) -> str:
    req = cur["request"] if cur else "(no task given yet)"
    return f"""{WORLD}

Bu Melati's current request to Axia: "{req}"
Axia just asked her: "{question}"

Answer as Bu Melati in 1-3 short sentences. Stay in character: confident and illogical, but include one genuinely useful hint about the nota (an item, a quantity or who it's for) if there is a task. Plain text only, no quotes around it.
Language: {LANG[lang]}"""


# ---- output checks shared by the generator, dataset builder and eval ----

def check_request(o) -> str | None:
    """Return None if `o` is a valid request object, else a short reason."""
    if not isinstance(o, dict):
        return "not an object"
    if not isinstance(o.get("request"), str) or not 15 <= len(o["request"]) <= 500:
        return "bad request text"
    if o.get("mood") not in MOODS:
        return "bad mood"
    if not isinstance(o.get("aside", ""), str) or not isinstance(o.get("axia"), str) or not o["axia"]:
        return "bad aside/axia"
    items = o.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 4:
        return "bad items"
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("name"), str) or not it["name"].strip():
            return "bad item"
        if not isinstance(it.get("qty"), int) or it["qty"] < 1:
            return "bad qty"
    if not isinstance(o.get("to"), str) or not o["to"].strip():
        return "bad to"
    return None


def check_judge(o) -> str | None:
    if not isinstance(o, dict):
        return "not an object"
    if o.get("verdict") not in ("approved", "rejected"):
        return "bad verdict"
    if not isinstance(o.get("reply"), str) or not 5 <= len(o["reply"]) <= 400:
        return "bad reply"
    if not isinstance(o.get("stamp"), str) or not o["stamp"].strip() or len(o["stamp"]) > 24:
        return "bad stamp"
    if not isinstance(o.get("patience"), int) or not -25 <= o["patience"] <= 5:
        return "bad patience"
    if o.get("mood") not in MOODS:
        return "bad mood"
    if not isinstance(o.get("axia"), str) or not o["axia"]:
        return "bad axia"
    return None


def check_ask(text) -> str | None:
    if not isinstance(text, str) or not 5 <= len(text.strip()) <= 400:
        return "bad length"
    if text.strip().startswith("{"):
        return "json instead of text"
    return None
