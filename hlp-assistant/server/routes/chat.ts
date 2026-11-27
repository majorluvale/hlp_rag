import { Router } from 'express';
import { config } from '../config';
import { all, get, now, parseJson, run, uid, type UserProfile } from '../db';
import { runAgent, type AgentEvent } from '../ai/agent';
import { applyUserFeedback } from '../ai/learning';
import { logout, publicUser, requestOtp, verifyOtp } from '../auth/auth';

export const chatRouter = Router();

// ---------- Auth (email + OTP, no password) ----------
chatRouter.get('/me', (req, res) => res.json({ user: publicUser(req.user) }));

chatRouter.post('/auth/request-otp', async (req, res) => {
  try {
    const host = String(req.headers.host ?? '').split(':')[0];
    const isLocal = ['localhost', '127.0.0.1', '[::1]'].includes(host) && !req.headers['x-forwarded-for'] && !req.headers['cf-connecting-ip'];
    const r = await requestOtp(req.body?.email, isLocal);
    res.status(r.status).json(r.body);
  } catch (e) {
    console.error('[otp]', e);
    res.status(500).json({ error: "Impossible d'envoyer l'email. Vérifiez la configuration SMTP." });
  }
});

chatRouter.post('/auth/verify-otp', (req, res) => {
  const r = verifyOtp(req, res, req.body?.email, req.body?.code);
  res.status(r.status).json(r.body);
});

chatRouter.post('/auth/logout', (req, res) => {
  logout(req, res);
  res.json({ ok: true });
});

const PROFILE_KEYS: (keyof UserProfile)[] = ['jobTitle', 'organisation', 'sector', 'country', 'expertise', 'language', 'interests'];

chatRouter.put('/me/profile', (req, res) => {
  const body = req.body ?? {};
  const current = parseJson<UserProfile>(req.user.profile_json, {});
  const next: UserProfile = { ...current };
  for (const k of PROFILE_KEYS) {
    if (typeof body.profile?.[k] === 'string') next[k] = body.profile[k].trim().slice(0, 300);
  }
  const name = typeof body.displayName === 'string' ? body.displayName.trim().slice(0, 80) : req.user.display_name;
  run('UPDATE users SET display_name = ?, profile_json = ? WHERE id = ?', name || null, JSON.stringify(next), req.user.id);
  const u = get('SELECT * FROM users WHERE id = ?', req.user.id);
  res.json({ user: publicUser(u) });
});

// ---------- Conversations ----------
chatRouter.get('/conversations', (req, res) => {
  res.json({
    conversations: all('SELECT id, title, updated_at AS updatedAt FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50', req.user.id),
  });
});

chatRouter.get('/conversations/:id', (req, res) => {
  const conv = get('SELECT id, title FROM conversations WHERE id = ? AND user_id = ?', req.params.id, req.user.id);
  if (!conv) return res.status(404).json({ error: 'Conversation introuvable' });
  const messages = all<{ id: string; role: string; content: string; trace_id: string | null; sources_json: string | null; created_at: number }>(
    'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at',
    req.params.id,
  );
  const votes = new Map(
    all<{ trace_id: string; rating: number }>(
      "SELECT trace_id, rating FROM feedback WHERE user_id = ? AND kind = 'user'",
      req.user.id,
    ).map((v) => [v.trace_id, v.rating]),
  );
  res.json({
    conversation: conv,
    messages: messages.map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content,
      traceId: m.trace_id,
      sources: parseJson(m.sources_json, []),
      vote: m.trace_id ? votes.get(m.trace_id) ?? 0 : 0,
      createdAt: m.created_at,
    })),
  });
});

chatRouter.delete('/conversations/:id', (req, res) => {
  run('DELETE FROM conversations WHERE id = ? AND user_id = ?', req.params.id, req.user.id);
  res.json({ ok: true });
});

// ---------- Chat (Server-Sent Events) ----------
const usage = new Map<string, number[]>();
function rateLimited(userId: string, isGuest: boolean) {
  const limit = isGuest ? config.guestHourlyLimit : config.guestHourlyLimit * 4;
  const hourAgo = Date.now() - 3600_000;
  const list = (usage.get(userId) ?? []).filter((t) => t > hourAgo);
  list.push(Date.now());
  usage.set(userId, list);
  return list.length > limit;
}

chatRouter.post('/chat', async (req, res) => {
  const question = String(req.body?.message ?? '').trim();
  if (!question) return res.status(400).json({ error: 'Message vide' });
  if (question.length > 4000) return res.status(400).json({ error: 'Message trop long (4000 caractères max.)' });
  if (rateLimited(req.user.id, Boolean(req.user.is_guest))) {
    return res.status(429).json({ error: 'Limite de questions atteinte pour cette heure. Connectez-vous pour une limite plus élevée.' });
  }

  let conversationId = typeof req.body?.conversationId === 'string' ? req.body.conversationId : null;
  if (conversationId && !get('SELECT id FROM conversations WHERE id = ? AND user_id = ?', conversationId, req.user.id)) conversationId = null;
  if (!conversationId) {
    conversationId = uid('c_');
    run('INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)', conversationId, req.user.id, question.slice(0, 80), now(), now());
  }
  const history = all<{ role: 'user' | 'assistant'; content: string }>(
    'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 10',
    conversationId,
  ).reverse();
  run('INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)', uid('m_'), conversationId, 'user', question, now());

  res.writeHead(200, {
    'Content-Type': 'text/event-stream; charset=utf-8',
    'Cache-Control': 'no-cache, no-transform',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  });
  const send = (e: AgentEvent | Record<string, unknown>) => {
    if (!res.writableEnded) res.write(`data: ${JSON.stringify(e)}\n\n`);
  };
  const ac = new AbortController();
  res.on('close', () => {
    if (!res.writableEnded) ac.abort();
  });
  const heartbeat = setInterval(() => res.write(': ping\n\n'), 15000);

  send({ type: 'conversation', conversationId });
  try {
    const result = await runAgent({ user: req.user, conversationId, question, history, emit: send, signal: ac.signal });
    const messageId = uid('m_');
    run(
      'INSERT INTO messages (id, conversation_id, role, content, trace_id, sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
      messageId, conversationId, 'assistant', result.answer, result.traceId, JSON.stringify(result.sources), now(),
    );
    run('UPDATE conversations SET updated_at = ? WHERE id = ?', now(), conversationId);
    send({ type: 'done', traceId: result.traceId, messageId });
  } catch (e) {
    console.error('[chat]', (e as Error).message);
    send({ type: 'error', error: 'Le service est momentanément indisponible. Réessayez dans un instant.', traceId: (e as { traceId?: string }).traceId });
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
});

chatRouter.post('/feedback', (req, res) => {
  const traceId = String(req.body?.traceId ?? '');
  const rating = Number(req.body?.rating) > 0 ? 1 : -1;
  const trace = get<{ user_id: string }>('SELECT user_id FROM traces WHERE id = ?', traceId);
  if (!trace || trace.user_id !== req.user.id) return res.status(404).json({ error: 'Réponse introuvable' });
  const comment = typeof req.body?.comment === 'string' ? req.body.comment.slice(0, 2000) : undefined;
  applyUserFeedback(traceId, req.user.id, rating, comment);
  res.json({ ok: true });
});

chatRouter.get('/info', (_req, res) => {
  const docs = get<{ n: number }>("SELECT COUNT(*) AS n FROM documents WHERE source_type = 'doc'")!.n;
  const web = get<{ n: number }>("SELECT COUNT(*) AS n FROM documents WHERE source_type = 'web'")!.n;
  const corrections = get<{ n: number }>('SELECT COUNT(*) AS n FROM corrections WHERE active = 1')!.n;
  res.json({ domains: config.web.domains, stats: { docs, web, corrections } });
});
