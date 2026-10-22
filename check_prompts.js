// Confirms prompts.py still matches the game's prompt builders.
// Usage: node check_prompts.js ../index.html  (then: python check_prompts.py ../index.html runs both and compares)
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const src = html.slice(html.indexOf("const WORLD ="), html.indexOf("/* ================= NOTA"));
const LANG = {
  id: html.match(/lang:"(Indonesian\.[^"]*)"/)[1],
  en: html.match(/lang:"(English,[^"]*)"/)[1],
};
const cases = JSON.parse(fs.readFileSync(0, "utf8"));
const out = cases.map(c => {
  const S = { lang: c.lang, day: c.day, done: (c.day - 1) * 4, recent: c.recent || [], cur: c.cur || null };
  const t = k => (k === "lang" ? LANG[S.lang] : k);
  const rp = n => (Number(n) || 0).toLocaleString("id-ID");
  const f = new Function("S", "t", "rp", src + "\nreturn { requestPrompt, judgePrompt, askPrompt };");
  const P = f(S, t, rp);
  if (c.kind === "request") return P.requestPrompt();
  if (c.kind === "judge") return P.judgePrompt(Object.assign({ total: c.nota.rows.reduce((s, r) => s + r.qty * r.price, 0) }, c.nota));
  return P.askPrompt(c.question);
});
process.stdout.write(JSON.stringify(out));
