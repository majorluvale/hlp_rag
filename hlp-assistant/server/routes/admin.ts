import { Router } from 'express';
import { all, get, parseJson, run } from '../db';
import { runAgent } from '../ai/agent';
import { applyReview } from '../ai/learning';
import { activatePrompt, createPromptVersion, listPrompts, PROMPT_NAMES, type PromptName } from '../ai/prompts';
import { requireRole } from '../auth/auth';
import { ingestAll, ingestStatus } from '../rag/ingest';
import { index } from '../rag/hybridIndex';

export const adminRouter = Router();
adminRouter.use(requireRole('reviewer', 'admin'));

// ---------- Dashboard ----------
adminRouter.get('/stats', (_req, res) => {
  const since = Date.now() - 30 * 24 * 3600_000;
  const one = <T>(sql: string, ...p: (string | number)[]) => get<T>(sql, ...p)!;
  const totals = one<{ n: number; lat: number; web: number; corr: number; err: number; tin: number; tout: number }>(
    `SELECT COUNT(*) n, AVG(latency_ms) lat, SUM(used_web) web, SUM(used_corrections) corr,
            SUM(status = 'error') err, SUM(input_tokens) tin, SUM(output_tokens) tout
     FROM traces WHERE is_playground = 0 AND created_at > ?`,
    since,
  );
  const votes = one<{ up: number; down: number }>(
    "SELECT SUM(rating > 0) up, SUM(rating < 0) down FROM feedback WHERE kind = 'user' AND created_at > ?",
    since,
  );
  const reviewer = one<{ avg: number; n: number }>("SELECT AVG(rating) avg, COUNT(*) n FROM feedback WHERE kind = 'reviewer'");
  const auto = one<{ avg: number; n: number }>("SELECT AVG(rating) avg, COUNT(*) n FROM feedback WHERE kind = 'auto' AND created_at > ?", since);
  const daily = all(
    `SELECT strftime('%Y-%m-%d', created_at / 1000, 'unixepoch') day, COUNT(*) n
     FROM traces WHERE is_playground = 0 AND created_at > ? GROUP BY day ORDER BY day`,
    since,
  );
  const byPrompt = all(
    `SELECT p.id, p.name, p.version, p.is_active active, COUNT(t.id) n,
            AVG(t.latency_ms) lat,
            (SELECT AVG(f.rating) FROM feedback f JOIN traces t2 ON t2.id = f.trace_id WHERE t2.prompt_version_id = p.id AND f.kind = 'auto') auto,
            (SELECT AVG(f.rating) FROM feedback f JOIN traces t2 ON t2.id = f.trace_id WHERE t2.prompt_version_id = p.id AND f.kind = 'reviewer') reviewer,
            (SELECT AVG(CASE WHEN f.rating > 0 THEN 1.0 ELSE 0 END) FROM feedback f JOIN traces t2 ON t2.id = f.trace_id WHERE t2.prompt_version_id = p.id AND f.kind = 'user') satisfaction
     FROM prompts p LEFT JOIN traces t ON t.prompt_version_id = p.id
     WHERE p.name = 'system' GROUP BY p.id ORDER BY p.version DESC`,
  );
  const users = one<{ guests: number; members: number }>(
    'SELECT SUM(is_guest) guests, SUM(1 - is_guest) members FROM users WHERE last_seen_at > ?',
    since,
  );
  const unreviewed = one<{ n: number }>(
    `SELECT COUNT(*) n FROM traces t WHERE is_playground = 0 AND status = 'ok'
     AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'reviewer')
     AND (EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND ((f.kind = 'user' AND f.rating < 0) OR (f.kind = 'auto' AND f.rating < 3.5))))`,
  );
  res.json({
    totals,
    votes,
    reviewer,
    auto,
    daily,
    byPrompt,
    users,
    toReview: unreviewed.n,
    corrections: one<{ n: number }>('SELECT COUNT(*) n FROM corrections WHERE active = 1').n,
    chunks: index.size,
  });
});

// ---------- Traces ----------
adminRouter.get('/traces', (req, res) => {
  const filter = String(req.query.filter ?? 'all');
  const q = String(req.query.q ?? '').trim();
  const where: string[] = ['t.is_playground = 0'];
  const params: (string | number)[] = [];
  if (filter === 'negative')
    where.push("EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND ((f.kind = 'user' AND f.rating < 0) OR (f.kind = 'auto' AND f.rating < 3.5)))");
  if (filter === 'unreviewed') where.push("NOT EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'reviewer')");
  if (filter === 'reviewed') where.push("EXISTS (SELECT 1 FROM feedback f WHERE f.trace_id = t.id AND f.kind = 'reviewer')");
  if (filter === 'web') where.push('t.used_web = 1');
  if (filter === 'error') where.push("t.status = 'error'");
  if (filter === 'playground') where.splice(0, 1, 't.is_playground = 1');
  if (q) {
    where.push('(t.question LIKE ? OR t.answer LIKE ?)');
    params.push(`%${q}%`, `%${q}%`);
  }
  const rows = all(
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

adminRouter.get('/traces/:id', (req, res) => {
  const trace = get(
    `SELECT t.*, u.display_name userName, u.email userEmail, u.is_guest isGuest, p.version promptVersion
     FROM traces t LEFT JOIN users u ON u.id = t.user_id LEFT JOIN prompts p ON p.id = t.prompt_version_id WHERE t.id = ?`,
    req.params.id,
  );
  if (!trace) return res.status(404).json({ error: 'Trace introuvable' });
  const spans = all('SELECT * FROM spans WHERE trace_id = ? ORDER BY start_at', req.params.id).map((s) => ({
    ...s,
    input: parseJson(s.input_json, s.input_json),
    output: parseJson(s.output_json, s.output_json),
    meta: parseJson(s.meta_json, null),
  }));
  const sources = all('SELECT * FROM trace_sources WHERE trace_id = ? ORDER BY n', req.params.id);
  const feedback = all(
    `SELECT f.*, u.display_name userName, u.email userEmail FROM feedback f LEFT JOIN users u ON u.id = f.user_id
     WHERE f.trace_id = ? ORDER BY f.created_at`,
    req.params.id,
  ).map((f) => ({ ...f, labels: parseJson(f.labels_json, null) }));
  res.json({ trace, spans, sources, feedback });
});

adminRouter.post('/traces/:id/review', async (req, res) => {
  const b = req.body ?? {};
  const rating = Math.min(5, Math.max(1, Math.round(Number(b.rating) || 0)));
  if (!rating) return res.status(400).json({ error: 'Note requise (1 à 5)' });
  try {
    const r = await applyReview({
      traceId: req.params.id,
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
adminRouter.get('/corrections', (_req, res) => {
  res.json({
    corrections: all(
      `SELECT c.id, c.trace_id traceId, c.question, c.corrected_answer correctedAnswer, c.active, c.hits, c.created_at createdAt,
              u.display_name author, u.email authorEmail
       FROM corrections c LEFT JOIN users u ON u.id = c.author_id ORDER BY c.created_at DESC`,
    ),
  });
});

adminRouter.patch('/corrections/:id', (req, res) => {
  if (typeof req.body?.active === 'boolean') run('UPDATE corrections SET active = ? WHERE id = ?', req.body.active, req.params.id);
  if (typeof req.body?.correctedAnswer === 'string') run('UPDATE corrections SET corrected_answer = ? WHERE id = ?', req.body.correctedAnswer, req.params.id);
  res.json({ ok: true });
});

// ---------- Prompts (versioned, editable by the team) ----------
adminRouter.get('/prompts', (_req, res) => res.json({ prompts: listPrompts(), names: PROMPT_NAMES }));

adminRouter.post('/prompts', (req, res) => {
  const name = String(req.body?.name) as PromptName;
  const content = String(req.body?.content ?? '');
  if (!PROMPT_NAMES.includes(name)) return res.status(400).json({ error: 'Prompt inconnu' });
  if (content.trim().length < 20) return res.status(400).json({ error: 'Prompt trop court' });
  if (name === 'system' && !content.includes('{{knowledge}}'))
    return res.status(400).json({ error: 'Le prompt système doit contenir la variable {{knowledge}}' });
  const row = createPromptVersion(name, content, String(req.body?.note ?? ''), req.user.email ?? req.user.id, Boolean(req.body?.activate));
  res.json({ prompt: row });
});

adminRouter.post('/prompts/:id/activate', (req, res) => {
  const row = activatePrompt(Number(req.params.id));
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
    });
    res.json(r);
  } catch (e) {
    res.status(500).json({ error: (e as Error).message });
  }
});

// ---------- Knowledge base ----------
adminRouter.get('/knowledge', (_req, res) => {
  res.json({
    documents: all(
      `SELECT d.id, d.title, d.source_type sourceType, d.url, d.pages, d.chunks, d.ingested_at ingestedAt,
              COALESCE(r.alpha, 1) alpha, COALESCE(r.beta, 1) beta,
              (SELECT COUNT(*) FROM trace_sources s WHERE s.doc_id = d.id AND s.cited = 1) citations
       FROM documents d LEFT JOIN source_rewards r ON r.doc_id = d.id ORDER BY d.source_type, d.title`,
    ),
    status: ingestStatus,
  });
});

adminRouter.post('/knowledge/reindex', (_req, res) => {
  void ingestAll();
  res.json({ ok: true, status: ingestStatus });
});

adminRouter.delete('/knowledge/:id', requireRole('admin'), (req, res) => {
  run('DELETE FROM documents WHERE id = ?', String(req.params.id));
  index.removeDoc(String(req.params.id));
  res.json({ ok: true });
});

// ---------- Users ----------
adminRouter.get('/users', (_req, res) => {
  res.json({
    users: all(
      `SELECT u.id, u.email, u.display_name displayName, u.role, u.is_guest isGuest, u.profile_json, u.created_at createdAt, u.last_seen_at lastSeenAt,
              (SELECT COUNT(*) FROM traces t WHERE t.user_id = u.id) questions
       FROM users u WHERE u.is_guest = 0 OR EXISTS (SELECT 1 FROM traces t WHERE t.user_id = u.id)
       ORDER BY u.last_seen_at DESC LIMIT 300`,
    ).map((u) => ({ ...u, profile: parseJson(u.profile_json, {}), profile_json: undefined })),
  });
});

adminRouter.patch('/users/:id', requireRole('admin'), (req, res) => {
  const role = String(req.body?.role);
  if (!['user', 'reviewer', 'admin'].includes(role)) return res.status(400).json({ error: 'Rôle invalide' });
  run('UPDATE users SET role = ? WHERE id = ? AND is_guest = 0', role, String(req.params.id));
  res.json({ ok: true });
});
