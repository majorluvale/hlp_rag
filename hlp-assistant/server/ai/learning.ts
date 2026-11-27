import { all, batch, get, now, q, run, toBlob, uid } from '../db';
import { embedOne } from '../rag/embeddings';
import { langfuseScore } from './tracing';

/**
 * Reinforcement from human feedback.
 * Each source document keeps a Beta(alpha, beta) reward. Positive feedback on an answer rewards
 * the sources it cited, negative feedback penalises them. Retrieval re-ranks with the posterior mean,
 * so useful documents rise and misleading ones sink over time.
 */
export async function updateRewards(docIds: string[], positive: boolean, weight: number) {
  const t = now();
  await batch(
    [...new Set(docIds)].map((id) =>
      q(
        `INSERT INTO source_rewards (doc_id, alpha, beta, updated_at) VALUES (?, ?, ?, ?)
         ON CONFLICT(doc_id) DO UPDATE SET alpha = alpha + ?, beta = beta + ?, updated_at = ?`,
        id, 1 + (positive ? weight : 0), 1 + (positive ? 0 : weight), t,
        positive ? weight : 0, positive ? 0 : weight, t,
      ),
    ),
  );
}

async function citedDocIds(traceId: string): Promise<string[]> {
  const rows = await all<{ doc_id: string | null; cited: number }>(
    'SELECT doc_id, cited FROM trace_sources WHERE trace_id = ? AND doc_id IS NOT NULL ORDER BY n',
    traceId,
  );
  const cited = rows.filter((r) => r.cited).map((r) => r.doc_id!);
  return cited.length ? cited : rows.slice(0, 3).map((r) => r.doc_id!);
}

export async function applyUserFeedback(traceId: string, userId: string, rating: 1 | -1, comment?: string) {
  const existing = await get<{ id: string; rating: number }>(
    "SELECT id, rating FROM feedback WHERE trace_id = ? AND user_id = ? AND kind = 'user'",
    traceId, userId,
  );
  const docs = await citedDocIds(traceId);
  if (existing) {
    await run('UPDATE feedback SET rating = ?, comment = COALESCE(?, comment) WHERE id = ?', rating, comment ?? null, existing.id);
    if (Number(existing.rating) !== rating) {
      // undo the previous vote, apply the new one
      await updateRewards(docs, Number(existing.rating) < 0, -0.5);
      await updateRewards(docs, rating > 0, 0.5);
    }
  } else {
    await run(
      "INSERT INTO feedback (id, trace_id, user_id, kind, rating, comment, created_at) VALUES (?, ?, ?, 'user', ?, ?, ?)",
      uid('fb_'), traceId, userId, rating, comment ?? null, now(),
    );
    await updateRewards(docs, rating > 0, 0.5);
  }
  langfuseScore(traceId, 'user_feedback', rating, comment);
}

export async function applyReview(input: {
  traceId: string;
  reviewerId: string;
  rating: number; // 1..5
  labels: string[];
  comment?: string;
  correctedAnswer?: string;
  badDocIds?: string[];
}) {
  const trace = await get<{ question: string }>('SELECT question FROM traces WHERE id = ?', input.traceId);
  if (!trace) throw new Error('Trace introuvable');

  await run(
    `INSERT INTO feedback (id, trace_id, user_id, kind, rating, labels_json, comment, corrected_answer, created_at)
     VALUES (?, ?, ?, 'reviewer', ?, ?, ?, ?, ?)`,
    uid('fb_'), input.traceId, input.reviewerId, input.rating, JSON.stringify(input.labels),
    input.comment ?? null, input.correctedAnswer ?? null, now(),
  );

  const docs = await citedDocIds(input.traceId);
  if (input.rating >= 4) await updateRewards(docs, true, 1.5);
  if (input.rating <= 2) await updateRewards(docs, false, 1.5);
  if (input.badDocIds?.length) await updateRewards(input.badDocIds, false, 3);

  let correctionId: string | null = null;
  if (input.correctedAnswer?.trim()) {
    const vec = await embedOne(trace.question, 'query');
    correctionId = uid('co_');
    await run(
      `INSERT INTO corrections (id, trace_id, question, corrected_answer, embedding, author_id, active, created_at)
       VALUES (?, ?, ?, ?, ?, ?, 1, ?)`,
      correctionId, input.traceId, trace.question, input.correctedAnswer.trim(), toBlob(vec), input.reviewerId, now(),
    );
  }
  langfuseScore(input.traceId, 'reviewer_rating', input.rating, input.comment);
  return { correctionId };
}
