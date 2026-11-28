import { Router } from 'express';
import { config } from '../config';
import { all, batch, dbKind, get, parseJson, q, run } from '../db';
import { runAgent } from '../ai/agent';
import { applyReview } from '../ai/learning';
import { makeClient } from '../ai/llm';
import { activatePrompt, createPromptVersion, listPrompts, PROMPT_NAMES, type PromptName } from '../ai/prompts';
import { requireRole } from '../auth/auth';
import { langOf } from '../i18n';
import { embeddingInfo } from '../rag/embeddings';
import { ingestAll, ingestStatus } from '../rag/ingest';
import { index, markIndexSynced } from '../rag/hybridIndex';
import { getLlmConfig, getSettings, maskSecret, setSettings } from '../settings';

export const adminRouter = Router();
adminRouter.use(requireRole('reviewer', 'admin'));

// ---------- Dashboard ----------
adminRouter.get('/stats', async (_req, res) => {
  const since = Date.now() - 30 * 24 * 3600_000;
  const [totals, votes, reviewer, auto, daily, byPrompt, users, unreviewed, corrections] = await Promise.all([
    get(
      `SELECT COUNT(*) n, AVG(latency_ms) lat, SUM(used_web) web, SUM(used_corrections) corr,
              SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) err, SUM(input_tokens) tin, SUM(output_tokens) tout
       FROM traces WHERE is_playground = 0 AND created_at > ?`,
      since,
    ),
    get(
      "SELECT SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) up, SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) down FROM feedback WHERE kind = 'user' AND created_at > ?",
      since,
    ),
    get("SELECT AVG(rating) avg, COUNT(*) n FROM feedback WHERE kind = 'reviewer'"),
    get("SELECT AVG(rating) avg, COUNT(*) n FROM feedback WHERE kind = 'auto' AND created_at > ?", since),
    all(
      `SELECT strftime('%Y-%m-%d', created_at / 1000, 'unixepoch') day, COUNT(*) n
       FROM traces WHERE is_playground = 0 AND created_at > ? GROUP BY day ORDER BY day`,
      since,
    ),
    all(
      `SELECT p.id, p.name, p.version, p.is_active active, COUNT(t.id) n,
              AVG(t.latency_ms) lat,
              (SELECT AVG(f.rating) FROM feedback f JOIN traces t2 ON t2.id = f.trace_id WHERE t2.prompt_version_id = p.id AND f.kind = 'auto') auto,
              (SELECT AVG(f.rating) FROM feedback f JOIN traces t2 ON t2.id = f.trace_id WHERE t2.prompt_version_id = p.id AND f.kind = 'reviewer') reviewer,
              (SELECT AVG(CASE WHEN f.rating > 0 THEN 1.0 ELSE 0 END) FROM feedback f JOIN traces t2 ON t2.id = f.trace_id WHERE t2.prompt_version_id = p.id AND f.kind = 'user') satisfaction
       FROM prompts p LEFT JOIN traces t ON t.prompt_version_id = p.id
       WHERE p.name = 'system' GROUP BY p.id ORDER BY p.version DESC`,
    ),
    get('SELECT SUM(is_guest) guests, SUM(1 - is_guest) members FROM users WHERE last_seen_at > ?', since),
    get(
      `SELECT COUNT(*) n FROM traces t WHERE is_playground = 0 AND status = 'ok'
       AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'reviewer')
       AND (EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND ((f.kind = 'user' AND f.rating < 0) OR (f.kind = 'auto' AND f.rating < 3.5))))`,
    ),
    get('SELECT COUNT(*) n FROM corrections WHERE active = 1'),
  ]);
  res.json({ totals, votes, reviewer, auto, daily, byPrompt, users, toReview: unreviewed.n, corrections: corrections.n, chunks: index.size });
});

// ---------- Traces ----------
adminRouter.get('/traces', async (req, res) => {
  const filter = String(req.query.filter ?? 'all');
  const search = String(req.query.q ?? '').trim();
  const where: string[] = ['t.is_playground = 0'];
  const params: (string | number)[] = [];
  if (filter === 'negative')
    where.push("EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND ((f.kind = 'user' AND f.rating < 0) OR (f.kind = 'auto' AND f.rating < 3.5)))");
  if (filter === 'unreviewed') where.push("NOT EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'reviewer')");
  if (filter === 'reviewed') where.push("EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'reviewer')");
  if (filter === 'web') where.push('t.used_web = 1');
  if (filter === 'error') where.push("t.status = 'error'");
  if (filter === 'playground') where.splice(0, 1, 't.is_playground = 1');
  if (search) {
    where.push('(t.question LIKE ? OR t.answer LIKE ?)');
    params.push(`%${search}%`, `%${search}%`);
  }
  const rows = await all(
    `SELECT t.id, t.question, substr(t.answer, 1, 220) answer, t.status, t.latency_ms latency, t.used_web usedWeb,
            t.used_corrections usedCorrections, t.created_at createdAt, t.prompt_version_id promptVersionId,
            u.display_name userName, u.email userEmail, u.is_guest isGuest,
            (SELECT rating FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'user' ORDER BY created_at DESC LIMIT 1) userVote,
            (SELECT rating FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'auto' ORDER BY created_at DESC LIMIT 1) autoScore,
            (SELECT rating FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'reviewer' ORDER BY created_at DESC LIMIT 1) reviewerScore
     FROM traces t LEFT JOIN users u ON u.id = t.user_id
     WHERE ${where.join(' AND ')} ORDER BY t.created_at DESC LIMIT 200`,
    ...params,
  );
  res.json({ traces: rows });
});

adminRouter.get('/traces/:id', async (req, res) => {
  const id = String(req.params.id);
  const trace = await get(
    `SELECT t.*, u.display_name userName, u.email userEmail, u.is_guest isGuest, p.version promptVersion
     FROM traces t LEFT JOIN users u ON u.id = t.user_id LEFT JOIN prompts p ON p.id = t.prompt_version_id WHERE t.id = ?`,
    id,
  );
  if (!trace) return res.status(404).json({ error: 'Trace introuvable' });
  const [spans, sources, feedback] = await Promise.all([
    all('SELECT * FROM spans WHERE trace_id = ? ORDER BY start_at', id),
    all('SELECT * FROM trace_sources WHERE trace_id = ? ORDER BY n', id),
    all(
      `SELECT f.*, u.display_name userName, u.email userEmail FROM feedback f LEFT JOIN users u ON u.id = f.user_id
       WHERE f.trace_id = ? ORDER BY f.created_at`,
      id,
    ),
  ]);
  res.json({
    trace,
    spans: spans.map((s) => ({
      ...s,
      input: parseJson(s.input_json, s.input_json),
      output: parseJson(s.output_json, s.output_json),
      meta: parseJson(s.meta_json, null),
    })),
    sources,
    feedback: feedback.map((f) => ({ ...f, labels: parseJson(f.labels_json, null) })),
  });
});

adminRouter.post('/traces/:id/review', async (req, res) => {
  const b = req.body ?? {};
  const rating = Math.min(5, Math.max(1, Math.round(Number(b.rating) || 0)));
  if (!rating) return res.status(400).json({ error: 'Note requise (1 à 5)' });
  try {
    const r = await applyReview({
      traceId: String(req.params.id),
      reviewerId: req.user.id,
      rating,
      labels: Array.isArray(b.labels) ? b.labels.map(String).slice(0, 10) : [],
      comment: typeof b.comment === 'string' ? b.comment.slice(0, 4000) : undefined,
      correctedAnswer: typeof b.correctedAnswer === 'string' ? b.correctedAnswer.slice(0, 12000) : undefined,
      badDocIds: Array.isArray(b.badDocIds) ? b.badDocIds.map(String) : [],
    });
    res.json({ ok: true, ...r });
  } catch (e) {
    res.status(400).json({ error: (e as Error).message });
  }
});

// ---------- Corrections memory ----------
adminRouter.get('/corrections', async (_req, res) => {
  res.json({
    corrections: await all(
      `SELECT c.id, c.trace_id traceId, c.question, c.corrected_answer correctedAnswer, c.active, c.hits, c.created_at createdAt,
              u.display_name author, u.email authorEmail
       FROM corrections c LEFT JOIN users u ON u.id = c.author_id ORDER BY c.created_at DESC`,
    ),
  });
});

adminRouter.patch('/corrections/:id', async (req, res) => {
  const id = String(req.params.id);
  if (typeof req.body?.active === 'boolean') await run('UPDATE corrections SET active = ? WHERE id = ?', req.body.active, id);
  if (typeof req.body?.correctedAnswer === 'string') await run('UPDATE corrections SET corrected_answer = ? WHERE id = ?', req.body.correctedAnswer, id);
  res.json({ ok: true });
});

// ---------- Prompts (versioned, editable by the team) ----------
adminRouter.get('/prompts', async (_req, res) => res.json({ prompts: await listPrompts(), names: PROMPT_NAMES }));

adminRouter.post('/prompts', async (req, res) => {
  const name = String(req.body?.name) as PromptName;
  const content = String(req.body?.content ?? '');
  if (!PROMPT_NAMES.includes(name)) return res.status(400).json({ error: 'Prompt inconnu' });
  if (content.trim().length < 20) return res.status(400).json({ error: 'Prompt trop court' });
  if (name === 'system' && !content.includes('{{knowledge}}'))
    return res.status(400).json({ error: 'Le prompt système doit contenir la variable {{knowledge}}' });
  const row = await createPromptVersion(name, content, String(req.body?.note ?? ''), req.user.email ?? req.user.id, Boolean(req.body?.activate));
  res.json({ prompt: row });
});

adminRouter.post('/prompts/:id/activate', async (req, res) => {
  const row = await activatePrompt(Number(req.params.id));
  if (!row) return res.status(404).json({ error: 'Version introuvable' });
  res.json({ ok: true });
});

adminRouter.post('/playground', async (req, res) => {
  const question = String(req.body?.question ?? '').trim();
  if (!question) return res.status(400).json({ error: 'Question requise' });
  try {
    const r = await runAgent({
      user: req.user,
      conversationId: null,
      question,
      history: [],
      promptOverride: typeof req.body?.content === 'string' && req.body.content.trim() ? req.body.content : undefined,
      playground: true,
      lang: langOf(req),
    });
    res.json(r);
  } catch (e) {
    res.status(500).json({ error: (e as Error).message });
  }
});

// ---------- Knowledge base ----------
adminRouter.get('/knowledge', async (_req, res) => {
  const settings = await getSettings();
  res.json({
    documents: await all(
      `SELECT d.id, d.title, d.source_type sourceType, d.url, d.pages, d.chunks, d.ingested_at ingestedAt,
              COALESCE(r.alpha, 1) alpha, COALESCE(r.beta, 1) beta,
              (SELECT COUNT(*) FROM trace_sources s WHERE s.doc_id = d.id AND s.cited = 1) citations
       FROM documents d LEFT JOIN source_rewards r ON r.doc_id = d.id WHERE d.chunks > 0 ORDER BY d.source_type, d.title`,
    ),
    status: ingestStatus,
    canIngest: !config.isVercel,
    embedding: embeddingInfo,
    indexEmbedding: settings.get('index.embedding') ?? null,
  });
});

adminRouter.post('/knowledge/reindex', (_req, res) => {
  if (config.isVercel) return res.status(400).json({ error: 'Sur Vercel, lancez « npm run ingest » depuis votre PC avec DATABASE_URL pointant vers Turso.' });
  void ingestAll();
  res.json({ ok: true, status: ingestStatus });
});

adminRouter.delete('/knowledge/:id', requireRole('admin'), async (req, res) => {
  const id = String(req.params.id);
  await batch([q('DELETE FROM chunks WHERE doc_id = ?', id), q('DELETE FROM documents WHERE id = ?', id)]);
  index.removeDoc(id);
  await markIndexSynced();
  res.json({ ok: true });
});

// ---------- Users ----------
adminRouter.get('/users', async (_req, res) => {
  const users = await all(
    `SELECT u.id, u.email, u.display_name displayName, u.role, u.is_guest isGuest, u.profile_json, u.created_at createdAt, u.last_seen_at lastSeenAt,
            (SELECT COUNT(*) FROM traces t WHERE t.user_id = u.id) questions
     FROM users u WHERE u.is_guest = 0 OR EXISTS (SELECT 1 FROM traces t WHERE t.user_id = u.id)
     ORDER BY u.last_seen_at DESC LIMIT 300`,
  );
  res.json({ users: users.map((u) => ({ ...u, profile: parseJson(u.profile_json, {}), profile_json: undefined })) });
});

adminRouter.patch('/users/:id', requireRole('admin'), async (req, res) => {
  const role = String(req.body?.role);
  if (!['user', 'reviewer', 'admin'].includes(role)) return res.status(400).json({ error: 'Rôle invalide' });
  await run('UPDATE users SET role = ? WHERE id = ? AND is_guest = 0', role, String(req.params.id));
  res.json({ ok: true });
});

// ---------- Settings: LLM API (admin only) ----------
async function publicLlmSettings() {
  const cfg = await getLlmConfig();
  return {
    provider: cfg.provider,
    baseURL: cfg.baseURL,
    model: cfg.model,
    fastModel: cfg.fastModel,
    temperature: cfg.temperature,
    maxSteps: cfg.maxSteps,
    reasoningEffort: cfg.reasoningEffort,
    hasApiKey: Boolean(cfg.apiKey),
    apiKeyHint: maskSecret(cfg.apiKey),
    source: cfg.source,
  };
}

adminRouter.get('/settings', requireRole('admin'), async (_req, res) => {
  res.json({
    llm: await publicLlmSettings(),
    system: { database: dbKind(), embedding: embeddingInfo, vercel: config.isVercel, langfuse: Boolean(config.langfuse.publicKey) },
  });
});

const cleanUrl = (u: unknown) => {
  const s = String(u ?? '').trim();
  if (!/^https?:\/\/[^\s]+$/i.test(s)) throw new Error('URL de base invalide (http:// ou https://)');
  return s.replace(/\/+$/, '');
};

adminRouter.put('/settings/llm', requireRole('admin'), async (req, res) => {
  try {
    const b = req.body ?? {};
    if (b.reset === true) {
      await setSettings(
        { 'llm.provider': null, 'llm.base_url': null, 'llm.api_key': null, 'llm.model': null, 'llm.fast_model': null, 'llm.temperature': null, 'llm.max_steps': null, 'llm.reasoning_effort': null },
        req.user.email ?? req.user.id,
      );
      return res.json({ llm: await publicLlmSettings() });
    }
    const model = String(b.model ?? '').trim();
    if (!model) throw new Error('Modèle requis');
    const temperature = Number(b.temperature);
    const maxSteps = Number(b.maxSteps);
    await setSettings(
      {
        'llm.provider': String(b.provider ?? 'custom').slice(0, 40),
        'llm.base_url': cleanUrl(b.baseURL),
        // empty = keep the saved key ; clearApiKey = remove it
        'llm.api_key': b.clearApiKey ? null : typeof b.apiKey === 'string' && b.apiKey.trim() ? b.apiKey.trim() : undefined,
        'llm.model': model.slice(0, 200),
        'llm.fast_model': String(b.fastModel ?? '').trim().slice(0, 200) || model.slice(0, 200),
        'llm.temperature': Number.isFinite(temperature) ? String(Math.min(2, Math.max(0, temperature))) : null,
        'llm.max_steps': Number.isFinite(maxSteps) ? String(Math.min(10, Math.max(1, Math.round(maxSteps)))) : null,
        'llm.reasoning_effort': ['low', 'medium', 'high', 'off'].includes(b.reasoningEffort) ? b.reasoningEffort : null,
      },
      req.user.email ?? req.user.id,
    );
    res.json({ llm: await publicLlmSettings() });
  } catch (e) {
    res.status(400).json({ error: (e as Error).message });
  }
});

/** Tests a configuration (the form values, or the saved key if the field is left empty) and lists available models. */
adminRouter.post('/settings/llm/test', requireRole('admin'), async (req, res) => {
  const saved = await getLlmConfig();
  const b = req.body ?? {};
  let baseURL: string;
  try {
    baseURL = b.baseURL ? cleanUrl(b.baseURL) : saved.baseURL;
  } catch (e) {
    return res.status(400).json({ ok: false, error: (e as Error).message });
  }
  const apiKey = typeof b.apiKey === 'string' && b.apiKey.trim() ? b.apiKey.trim() : baseURL === saved.baseURL ? saved.apiKey : '';
  const model = String(b.model ?? '').trim() || saved.model;
  const client = makeClient(baseURL, apiKey);
  let models: string[] = [];
  try {
    const list = await client.models.list();
    models = list.data.map((m) => m.id).sort();
  } catch {
    /* some providers do not expose /models */
  }
  const t0 = Date.now();
  try {
    const r = await client.chat.completions.create({
      model,
      messages: [{ role: 'user', content: 'Réponds uniquement par le mot OK.' }],
      max_tokens: 200,
    });
    res.json({ ok: true, latencyMs: Date.now() - t0, reply: (r.choices[0]?.message?.content ?? '').slice(0, 100), models });
  } catch (e) {
    res.json({ ok: false, error: (e as Error).message.slice(0, 400), models });
  }
});
