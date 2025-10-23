/**
 * Budget-Aware Orchestrator with Three Allocation Strategies
 * ==========================================================
 * 
 * Three conditions for fairness evaluation:
 * 1. UNIFORM: Fixed top_k=5 for all (baseline)
 * 2. LANGUAGE_AWARE: top_k=8 for MI, top_k=5 for EN (language fairness)
 * 3. FAIRNESS_AWARE: top_k=8 for MI OR complex, top_k=5 otherwise (full fairness)
 */

import Fastify from "fastify";
import fs from "node:fs";
import path from "node:path";

const PORT = Number(process.env.PORT || 8000);
const RETRIEVER_BASE = process.env.RETRIEVER_BASE || "http://localhost:8001";

const log = Fastify({
  logger: { level: "info" }
});

// Health check
log.get("/", async () => ({ 
  ok: true, 
  service: "budget-aware-orchestrator", 
  version: "1.0.0",
  modes: ["uniform", "language_aware", "fairness_aware"]
}));

/**
 * Request body type
 */
type QueryBody = {
  query: string;
  lang?: "en" | "mi";
  complexity?: "simple" | "complex";
  mode?: "uniform" | "language_aware" | "fairness_aware";
  use_rerank?: boolean;
};

/**
 * Helper for JSON POST requests
 */
async function postJSON(url: string, body: any) {
  const t0 = Date.now();
  const res = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${url} -> ${res.status}: ${text}`);
  }
  const json = await res.json();
  const ms = Date.now() - t0;
  return { json, ms };
}

/**
 * Budget allocation planner with three strategies
 */
function planBudget(body: QueryBody) {
  const mode = body.mode || "uniform";
  const lang = body.lang || "en";
  const complexity = body.complexity || "simple";
  const use_rerank = body.use_rerank !== false;
  
  let top_k: number;
  let rerank_k: number = use_rerank ? 3 : 0;
  let rationale: string;
  
  switch (mode) {
    case "uniform":
      // CONDITION 1: Equal budget for everyone
      top_k = 5;
      rationale = "Fixed budget (top_k=5) for all queries";
      break;
      
    case "language_aware":
      // CONDITION 2: More budget for Māori (lower-resourced language)
      if (lang === "mi") {
        top_k = 8;
        rationale = "Increased budget for Te Reo Māori (lower-resourced language)";
      } else {
        top_k = 5;
        rationale = "Standard budget for English";
      }
      break;
      
    case "fairness_aware":
      // CONDITION 3: More budget for Māori OR complex queries
      if (lang === "mi" && complexity === "complex") {
        top_k = 8;
        rationale = "Increased budget: Māori + complex query (double vulnerability)";
      } else if (lang === "mi") {
        top_k = 8;
        rationale = "Increased budget: Te Reo Māori (language vulnerability)";
      } else if (complexity === "complex") {
        top_k = 8;
        rationale = "Increased budget: Complex query (retrieval difficulty)";
      } else {
        top_k = 5;
        rationale = "Standard budget: English + simple query";
      }
      break;
      
    default:
      top_k = 5;
      rationale = "Unknown mode, using default";
  }
  
  return { 
    top_k, 
    rerank_k,
    mode,
    lang,
    complexity,
    rationale
  };
}

/**
 * Main query endpoint
 */
log.post("/query", async (req, reply) => {
  const body = req.body as QueryBody;
  const started = Date.now();
  
  // Validate input
  if (!body.query) {
    return reply.code(400).send({ error: "Missing 'query' field" });
  }
  
  // Plan budget allocation
  const plan = planBudget(body);
  
  log.log.info(`[${plan.mode}] Query: ${body.query.substring(0, 50)}...`);
  log.log.info(`[${plan.mode}] Plan: top_k=${plan.top_k}, rerank=${plan.rerank_k}, lang=${plan.lang}, complexity=${plan.complexity}`);

  try {
    // Stage 1: Retrieve candidates
    const { json: candidates, ms: ms_retrieve } = await postJSON(
      `${RETRIEVER_BASE}/retrieve`, 
      {
        query: body.query,
        top_k: plan.top_k
      }
    );

    if (!Array.isArray(candidates) || candidates.length === 0) {
      log.log.warn(`[${plan.mode}] No candidates retrieved`);
    }

    // Stage 2: Optional reranking
    let passages = candidates;
    let ms_rerank = 0;
    
    if (plan.rerank_k > 0 && Array.isArray(candidates) && candidates.length > 0) {
      try {
        const { json: reranked, ms } = await postJSON(
          `${RETRIEVER_BASE}/rerank`,
          {
            query: body.query,
            candidates: candidates,
            k: plan.rerank_k
          }
        );
        passages = reranked;
        ms_rerank = ms;
      } catch (err: any) {
        log.log.error(`[${plan.mode}] Rerank failed: ${err.message}, using candidates`);
        passages = candidates;
      }
    }

    const total_ms = Date.now() - started;

    // Build response record
    const record = {
      ts: new Date().toISOString(),
      query: body.query,
      lang: plan.lang,
      complexity: plan.complexity,
      mode: plan.mode,
      plan: {
        top_k: plan.top_k,
        rerank_k: plan.rerank_k,
        rationale: plan.rationale
      },
      metrics: {
        retrieve_ms: ms_retrieve,
        rerank_ms: ms_rerank,
        total_ms: total_ms,
        num_candidates: candidates.length,
        num_passages: passages.length
      },
      passages: passages
    };

    // Log to trace file
    const tracesDir = path.resolve(process.cwd(), "../../traces");
    try {
      fs.mkdirSync(tracesDir, { recursive: true });
      fs.appendFileSync(
        path.join(tracesDir, "orchestrator_requests.jsonl"),
        JSON.stringify(record) + "\n"
      );
    } catch (err: any) {
      log.log.error(`Failed to write trace: ${err.message}`);
    }

    return reply.send(record);
    
  } catch (err: any) {
    const total_ms = Date.now() - started;
    log.log.error(`[${plan.mode}] Error: ${err.message}`);
    
    const errorRecord = {
      ts: new Date().toISOString(),
      query: body.query,
      lang: plan.lang,
      complexity: plan.complexity,
      mode: plan.mode,
      plan: {
        top_k: plan.top_k,
        rerank_k: plan.rerank_k,
        rationale: plan.rationale
      },
      metrics: {
        total_ms: total_ms
      },
      error: err.message,
      passages: []
    };
    
    return reply.code(500).send(errorRecord);
  }
});

/**
 * Test endpoint for quick health checks
 */
log.get("/test", async (req, reply) => {
  return reply.send({
    ok: true,
    retriever: RETRIEVER_BASE,
    timestamp: new Date().toISOString()
  });
});

// Start server
log.listen({ port: PORT, host: "0.0.0.0" }).then(() => {
  console.log(`✓ Budget-Aware Orchestrator listening on port ${PORT}`);
  console.log(`✓ Retriever service: ${RETRIEVER_BASE}`);
  console.log(`✓ Available modes: uniform, language_aware, fairness_aware`);
});
