import Fastify from "fastify";
import fs from "node:fs";
import path from "node:path";

const PORT = Number(process.env.PORT || 8000);
const RETRIEVER_BASE = process.env.RETRIEVER_BASE || "http://localhost:8001";

const log = Fastify({
  logger: { level: "info" }
});

log.get("/", async () => ({ ok: true, service: "orchestrator", endpoint: "/query" }));

type QueryBody = {
  query: string;
  lang?: "en" | "mi";
  mode?: "baseline" | "budgeter";
  use_rerank?: boolean;
};

// tiny helper
async function postJSON(url: string, body: any) {
  const t0 = Date.now();
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  const json = await res.json();
  const ms = Date.now() - t0;
  return { json, ms };
}

function plan(body: QueryBody) {
  // Right-sized planner: slightly deeper for Māori; rerank on for both when asked
  const top_k_base = body.lang === "mi" ? 8 : 6;
  const rerank_k = body.use_rerank === false ? 0 : (body.lang === "mi" ? 5 : 3);
  return { top_k: top_k_base, rerank_k };
}

log.post("/query", async (req, reply) => {
  const body = req.body as QueryBody;
  const started = Date.now();
  const p = plan(body);

  try {
    // Stage 1: retrieve
    const { json: cands, ms: ms_retr } = await postJSON(`${RETRIEVER_BASE}/retrieve`, {
      query: body.query,
      top_k: p.top_k
    });

    // Stage 2: optional re-rank
    let used = cands;
    let ms_rer = 0;
    if (p.rerank_k > 0 && Array.isArray(cands) && cands.length > 0) {
      const { json: rer, ms } = await postJSON(`${RETRIEVER_BASE}/rerank`, {
        query: body.query,
        candidates: cands,
        k: p.rerank_k
      });
      used = rer;
      ms_rer = ms;
    }

    // Stub LLM answer: echo best passage; (swap in cloud LLM later)
    const best = Array.isArray(used) && used.length ? used[0] : null;
    const answer = best
      ? `According to ${best.doc_id}: ${best.text}`
      : "I couldn't find a grounded passage to answer that.";
    const citations = best
      ? [{ doc_id: best.doc_id, char_start: best.char_start, char_end: best.char_end }]
      : [];

    const total_ms = Date.now() - started;
    const record = {
      ts: new Date().toISOString(),
      query: body.query,
      lang: body.lang || "en",
      mode: body.mode || "baseline",
      use_rerank: body.use_rerank !== false,
      plan: p,
      metrics: { retrieve_ms: ms_retr, rerank_ms: ms_rer, total_ms, cost_usd: 0.0 },
      response: { answer, citations, refusal: false }
    };

    // write JSONL trace
    const tracesDir = path.resolve(process.cwd(), "../../traces");
    fs.mkdirSync(tracesDir, { recursive: true });
    fs.appendFileSync(path.join(tracesDir, "requests.jsonl"), JSON.stringify(record) + "\n");

    return reply.send(record);
  } catch (err: any) {
    const total_ms = Date.now() - started;
    const errorRec = {
      ts: new Date().toISOString(),
      query: body.query,
      lang: body.lang || "en",
      mode: body.mode || "baseline",
      use_rerank: body.use_rerank !== false,
      plan: p,
      metrics: { total_ms, cost_usd: 0.0 },
      response: { answer: "", citations: [], refusal: true },
      error: String(err?.message || err)
    };
    return reply.code(500).send(errorRec);
  }
});

log.listen({ port: PORT, host: "0.0.0.0" }).then(() => {
  log.log.info(`Server listening on ${PORT}`);
});
