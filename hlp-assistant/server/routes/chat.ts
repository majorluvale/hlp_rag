import { Router } from 'express';
import { config } from '../config';
import { all, batch, get, now, parseJson, q, run, uid, type UserProfile, type UserRow } from '../db';
import { runAgent, type AgentEvent } from '../ai/agent';
import { applyUserFeedback } from '../ai/learning';
import { logout, publicUser, requestOtp, verifyOtp } from '../auth/auth';

export const chatRouter = Router();

// ---------- Auth (email + OTP, no password) ----------
chatRouter.get('/me', (req, res) => res.json({ user: publicUser(req.user) }));

chatRouter.post('/auth/request-otp', async (req, res) => {
  try {
    const host = String(req.headers.host ?? '').split(':')[0];
    const isLocal =
      ['localhost', '127.0.0.1', '[::1]'].includes(host) && !req.headers['x-forwarded-for'] && !req.headers['cf-connecting-ip'] && !config.isVercel;
    const r = await requestOtp(req.body?.email, isLocal);
    res.status(r.status).json(r.body);
  } catch (e) {
    console.error('[otp]', e);
    res.status(500).json({ error: "Impossible d'envoyer l'email. Vérifiez la configuration SMTP." });
  }
});

chatRouter.post('/auth/verify-otp', async (req, res) => {
  const r = await verifyOtp(req, res, req.body?.email, req.body?.code);
  res.status(r.status).json(r.body);
});

chatRouter.post('/auth/logout', async (req, res) => {
  await logout(req, res);
  res.json({ ok: true });
});

const PROFILE_KEYS: (keyof UserProfile)[] = ['jobTitle', 'organisation', 'sector', 'country', 'expertise', 'language', 'interests'];

chatRouter.put('/me/profile', async (req, res) => {
  const body = req.body ?? {};
  const current = parseJson<UserProfile>(req.user.profile_json, {});
  const next: UserProfile = { ...current };
  for (const k of PROFILE_KEYS) {
    if (typeof body.profile?.[k] === 'string') next[k] = body.profile[k].trim().slice(0, 300);
  }
  const name = typeof body.displayName === 'string' ? body.displayName.trim().slice(0, 80) : req.user.display_name;
  await run('UPDATE users SET display_name = ?, profile_json = ? WHERE id = ?', name || null, JSON.stringify(next), req.user.id);
  const u = await get<UserRow>('SELECT * FROM users WHERE id = ?', req.user.id);
  res.json({ user: publicUser(u!) });
});

// ---------- Conversations ----------
chatRouter.get('/conversations', async (req, res) => {
  res.json({
    conversations: await all(
      'SELECT id, title, updated_at AS updatedAt FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50',
      req.user.id,
    ),
  });
});

chatRouter.get('/conversations/:id', async (req, res) => {
  const id = String(req.params.id);
  const conv = await get('SELECT id, title FROM conversations WHERE id = ? AND user_id = ?', id, req.user.id);
  if (!conv) return res.status(404).json({ error: 'Conversation introuvable' });
  const [messages, votes] = await Promise.all([
    all<{ id: string; role: string; content: string; trace_id: string | null; sources_json: string | null; created_at: number }>(
      'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at',
      id,
    ),
    all<{ trace_id: string; rating: number }>("SELECT trace_id, rating FROM feedback WHERE user_id = ? AND kind = 'user'", req.user.id),
  ]);
  const voteMap = new Map(votes.map((v) => [v.trace_id, Number(v.rating)]));
  res.json({
    conversation: conv,
    messages: messages.map((m) => ({
      id: m.id,
      role: m.role,
      content: m.content,
      traceId: m.trace_id,
      sources: parseJson(m.sources_json, []),
      vote: m.trace_id ? voteMap.get(m.trace_id) ?? 0 : 0,
      createdAt: m.created_at,
    })),
  });
});

chatRouter.delete('/conversations/:id', async (req, res) => {
  const id = String(req.params.id);
  if (await get('SELECT id FROM conversations WHERE id = ? AND user_id = ?', id, req.user.id)) {
    await batch([q('DELETE FROM messages WHERE conversation_id = ?', id), q('DELETE FROM conversations WHERE id = ?', id)]);
  }
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
  if (conversationId && !(await get('SELECT id FROM conversations WHERE id = ? AND user_id = ?', conversationId, req.user.id))) conversationId = null;
  if (!conversationId) {
    conversationId = uid('c_');
    await run('INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)', conversationId, req.user.id, question.slice(0, 80), now(), now());
  }
  const history = (
    await all<{ role: 'user' | 'assistant'; content: string }>(
      'SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 10',
      conversationId,
    )
  ).reverse();
  await run('INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)', uid('m_'), conversationId, 'user', question, now());

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
    await batch([
      q(
        'INSERT INTO messages (id, conversation_id, role, content, trace_id, sources_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
        messageId, conversationId, 'assistant', result.answer, result.traceId, JSON.stringify(result.sources), now(),
      ),
      q('UPDATE conversations SET updated_at = ? WHERE id = ?', now(), conversationId),
    ]);
    send({ type: 'done', traceId: result.traceId, messageId });
  } catch (e) {
    const err = e as Error & { traceId?: string; userFacing?: boolean };
    console.error('[chat]', err.message);
    send({
      type: 'error',
      error: err.userFacing ? err.message : 'Le service est momentanément indisponible. Réessayez dans un instant.',
      traceId: err.traceId,
    });
  } finally {
    clearInterval(heartbeat);
    res.end();
  }
});

chatRouter.post('/feedback', async (req, res) => {
  const traceId = String(req.body?.traceId ?? '');
  const rating = Number(req.body?.rating) > 0 ? 1 : -1;
  const trace = await get<{ user_id: string }>('SELECT user_id FROM traces WHERE id = ?', traceId);
  if (!trace || trace.user_id !== req.user.id) return res.status(404).json({ error: 'Réponse introuvable' });
  const comment = typeof req.body?.comment === 'string' ? req.body.comment.slice(0, 2000) : undefined;
  await applyUserFeedback(traceId, req.user.id, rating, comment);
  res.json({ ok: true });
});

chatRouter.get('/info', async (_req, res) => {
  const r = await get<{ docs: number; web: number; corrections: number }>(
    `SELECT (SELECT COUNT(*) FROM documents WHERE source_type = 'doc' AND chunks > 0) AS docs,
            (SELECT COUNT(*) FROM documents WHERE source_type = 'web' AND chunks > 0) AS web,
            (SELECT COUNT(*) FROM corrections WHERE active = 1) AS corrections`,
  );
  res.json({ domains: config.web.domains, stats: r });
});
