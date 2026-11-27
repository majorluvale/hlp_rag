import { all, fromBlob } from '../db';
import { cosine, embedOne } from './embeddings';
import { ensureFreshIndex, index } from './hybridIndex';

export interface Hit {
  chunkId: number;
  docId: string;
  title: string;
  page: number | null;
  url: string | null;
  sourceType: string;
  text: string;
  score: number;
  cosine: number;
  reward: number;
}

export interface CorrectionHit {
  id: string;
  question: string;
  correctedAnswer: string;
  similarity: number;
}

/** Posterior mean of the Beta(alpha, beta) reward of each source, learned from feedback. */
export async function rewardMap(): Promise<Map<string, number>> {
  const rows = await all<{ doc_id: string; alpha: number; beta: number }>('SELECT doc_id, alpha, beta FROM source_rewards');
  return new Map(rows.map((r) => [r.doc_id, r.alpha / (r.alpha + r.beta)]));
}

const MAX_PER_DOC = 3;

/**
 * Hybrid retrieval (dense + BM25 + RRF), re-ranked by the reward each source has earned
 * from user and reviewer feedback (reinforcement signal), with per-document diversity.
 */
export async function retrieve(query: string, k = 6, qvec?: Float32Array) {
  await ensureFreshIndex();
  const [qv, rewards] = await Promise.all([qvec ? Promise.resolve(qvec) : embedOne(query, 'query'), rewardMap()]);
  const candidates = index.search(qv, query, 40);

  const scored: Hit[] = candidates.map((c) => {
    const reward = rewards.get(c.chunk.docId) ?? 0.5;
    // reward 0.5 (neutral prior) => factor 1.0 ; range 0.6 .. 1.4
    const score = c.rrf * (0.6 + 0.8 * reward) * (0.5 + c.cosine);
    return {
      chunkId: c.chunk.id,
      docId: c.chunk.docId,
      title: c.chunk.title,
      page: c.chunk.page,
      url: c.chunk.url,
      sourceType: c.chunk.sourceType,
      text: c.chunk.text,
      score,
      cosine: c.cosine,
      reward,
    };
  });
  scored.sort((a, b) => b.score - a.score);

  const perDoc = new Map<string, number>();
  const hits: Hit[] = [];
  for (const h of scored) {
    const n = perDoc.get(h.docId) ?? 0;
    if (n >= MAX_PER_DOC) continue;
    perDoc.set(h.docId, n + 1);
    hits.push(h);
    if (hits.length >= k) break;
  }
  return { hits, qvec: qv };
}

const CORRECTION_THRESHOLD = 0.86;

/** Reviewer-validated answers to similar questions: injected with top priority in the prompt. */
export async function findCorrections(qvec: Float32Array, limit = 3): Promise<CorrectionHit[]> {
  const rows = await all<{ id: string; question: string; corrected_answer: string; embedding: Uint8Array }>(
    'SELECT id, question, corrected_answer, embedding FROM corrections WHERE active = 1',
  );
  return rows
    .map((r) => ({
      id: r.id,
      question: r.question,
      correctedAnswer: r.corrected_answer,
      similarity: cosine(qvec, fromBlob(r.embedding)),
    }))
    .filter((r) => r.similarity >= CORRECTION_THRESHOLD)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, limit);
}
