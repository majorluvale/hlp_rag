import { background } from '../background';
import { config } from '../config';
import { batch, now, q, run, uid, type Stmt } from '../db';

const MAX_JSON = 60000;
const clip = (v: unknown) => {
  if (v === undefined) return null;
  const s = JSON.stringify(v);
  return s.length > MAX_JSON ? s.slice(0, MAX_JSON) + '…"[tronqué]"' : s;
};

interface SpanRecord {
  id: string;
  type: string;
  name: string;
  input: unknown;
  output?: unknown;
  meta?: Record<string, unknown>;
  error?: string;
  start: number;
  duration: number;
}

export interface TraceSource {
  n: number;
  kind: 'doc' | 'web' | 'correction';
  docId?: string | null;
  chunkId?: number | null;
  title: string;
  url?: string | null;
  page?: number | null;
  score?: number | null;
  snippet: string;
  cited?: boolean;
}

export interface TraceInfo {
  userId: string | null;
  conversationId: string | null;
  question: string;
  userContext: string;
  promptVersionId: number | null;
  model: string;
  playground?: boolean;
}

/**
 * Records every step of an answer (retrieval, LLM generations, tool calls) so that reviewers
 * can inspect and grade it in the console. Optionally mirrored to Langfuse.
 */
export class Tracer {
  readonly id = uid('tr_');
  private readonly startedAt = now();
  private spans: SpanRecord[] = [];
  usage = { input: 0, output: 0 };
  usedWeb = false;
  usedCorrections = false;

  private constructor(private readonly info: TraceInfo) {}

  static async start(info: TraceInfo) {
    const t = new Tracer(info);
    await run(
      `INSERT INTO traces (id, user_id, conversation_id, question, user_context, prompt_version_id, model, status, is_playground, created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)`,
      t.id, info.userId, info.conversationId, info.question, info.userContext, info.promptVersionId, info.model,
      info.playground ? 1 : 0, t.startedAt,
    );
    return t;
  }

  span(type: string, name: string, input: unknown) {
    const rec: SpanRecord = { id: uid('sp_'), type, name, input, start: now(), duration: 0 };
    const save = () => {
      rec.duration = now() - rec.start;
      this.spans.push(rec);
    };
    return {
      end: (output: unknown, meta?: Record<string, unknown>) => {
        rec.output = output;
        rec.meta = meta;
        save();
      },
      fail: (err: unknown, meta?: Record<string, unknown>) => {
        rec.error = err instanceof Error ? err.message : String(err);
        rec.meta = meta;
        save();
      },
    };
  }

  addUsage(u: { prompt_tokens?: number; completion_tokens?: number } | null | undefined) {
    if (!u) return;
    this.usage.input += u.prompt_tokens ?? 0;
    this.usage.output += u.completion_tokens ?? 0;
  }

  /** Writes spans, sources and the final answer in one batch (one round trip on Turso). */
  async finish(result: { answer: string; sources: TraceSource[]; error?: string }) {
    const latency = now() - this.startedAt;
    const stmts: Stmt[] = [
      q(
        `UPDATE traces SET answer = ?, status = ?, error = ?, latency_ms = ?, input_tokens = ?, output_tokens = ?, used_web = ?, used_corrections = ? WHERE id = ?`,
        result.answer, result.error ? 'error' : 'ok', result.error, latency, this.usage.input, this.usage.output,
        this.usedWeb ? 1 : 0, this.usedCorrections ? 1 : 0, this.id,
      ),
      ...this.spans.map((s) =>
        q(
          `INSERT INTO spans (id, trace_id, type, name, input_json, output_json, meta_json, error, start_at, duration_ms)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          s.id, this.id, s.type, s.name, clip(s.input), clip(s.output), clip(s.meta), s.error, s.start, s.duration,
        ),
      ),
      ...result.sources.map((s) =>
        q(
          `INSERT OR REPLACE INTO trace_sources (trace_id, n, kind, doc_id, chunk_id, title, url, page, score, cited, snippet)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
          this.id, s.n, s.kind, s.docId, s.chunkId, s.title, s.url, s.page, s.score, s.cited ? 1 : 0, s.snippet.slice(0, 1200),
        ),
      ),
    ];
    await batch(stmts);
    background(mirrorToLangfuse(this.id, this.info, this.spans, result, latency));
  }
}

// ---------- Langfuse (optional) ----------
const lfEnabled = () => Boolean(config.langfuse.publicKey && config.langfuse.secretKey);

async function lfPost(batchEvents: unknown[]) {
  if (!lfEnabled()) return;
  const auth = Buffer.from(`${config.langfuse.publicKey}:${config.langfuse.secretKey}`).toString('base64');
  const res = await fetch(`${config.langfuse.baseUrl.replace(/\/$/, '')}/api/public/ingestion`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Basic ${auth}` },
    body: JSON.stringify({ batch: batchEvents }),
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) console.warn('[langfuse]', res.status);
}

async function mirrorToLangfuse(traceId: string, info: TraceInfo, spans: SpanRecord[], result: { answer: string; error?: string }, latency: number) {
  if (!lfEnabled()) return;
  const ts = new Date().toISOString();
  const iso = (ms: number) => new Date(ms).toISOString();
  const events: unknown[] = [
    {
      id: uid('ev_'),
      type: 'trace-create',
      timestamp: ts,
      body: {
        id: traceId,
        name: info.playground ? 'hlp-playground' : 'hlp-chat',
        userId: info.userId,
        sessionId: info.conversationId,
        input: info.question,
        output: result.answer,
        metadata: { userContext: info.userContext, promptVersionId: info.promptVersionId, latency, error: result.error },
      },
    },
  ];
  for (const s of spans) {
    const isLlm = s.type === 'llm';
    events.push({
      id: uid('ev_'),
      type: isLlm ? 'generation-create' : 'span-create',
      timestamp: ts,
      body: {
        id: s.id,
        traceId,
        name: s.name,
        startTime: iso(s.start),
        endTime: iso(s.start + s.duration),
        input: s.input,
        output: s.output,
        metadata: s.meta,
        level: s.error ? 'ERROR' : 'DEFAULT',
        statusMessage: s.error,
        ...(isLlm ? { model: info.model, usage: s.meta?.usage } : {}),
      },
    });
  }
  await lfPost(events);
}

export function langfuseScore(traceId: string, name: string, value: number, comment?: string) {
  background(
    lfPost([
      {
        id: uid('ev_'),
        type: 'score-create',
        timestamp: new Date().toISOString(),
        body: { id: uid('sc_'), traceId, name, value, comment },
      },
    ]),
  );
}
