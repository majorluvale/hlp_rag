import { all, fromBlob, get } from '../db';
import { tokenize } from './text';

export interface IndexedChunk {
  id: number;
  docId: string;
  title: string;
  page: number | null;
  url: string | null;
  sourceType: string;
  text: string;
  vec: Float32Array;
  tf: Map<string, number>;
  len: number;
}

export interface Candidate {
  chunk: IndexedChunk;
  rrf: number;
  cosine: number;
  vecRank: number | null;
  bm25Rank: number | null;
}

const K1 = 1.4;
const B = 0.75;
const RRF_K = 60;

/**
 * In-memory hybrid index: dense vectors (cosine) + sparse BM25, fused with Reciprocal Rank Fusion.
 * Persisted in SQLite, loaded at startup. Brute force is fast up to ~100k chunks.
 */
class HybridIndex {
  private chunks: IndexedChunk[] = [];
  private df = new Map<string, number>();
  private totalLen = 0;

  get size() {
    return this.chunks.length;
  }

  add(c: Omit<IndexedChunk, 'tf' | 'len'>) {
    const tokens = tokenize(c.title + ' ' + c.text);
    const tf = new Map<string, number>();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const t of tf.keys()) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    this.totalLen += tokens.length;
    this.chunks.push({ ...c, tf, len: tokens.length });
  }

  /** Removes a document's chunks, updating the BM25 statistics in place (no re-tokenisation). */
  removeDoc(docId: string) {
    if (!this.chunks.some((c) => c.docId === docId)) return;
    const keep: IndexedChunk[] = [];
    for (const c of this.chunks) {
      if (c.docId !== docId) {
        keep.push(c);
        continue;
      }
      this.totalLen -= c.len;
      for (const t of c.tf.keys()) {
        const n = (this.df.get(t) ?? 1) - 1;
        if (n > 0) this.df.set(t, n);
        else this.df.delete(t);
      }
    }
    this.chunks = keep;
  }

  clear() {
    this.chunks = [];
    this.df.clear();
    this.totalLen = 0;
  }

  private vectorSearch(q: Float32Array, k: number) {
    const scored: { i: number; s: number }[] = [];
    for (let i = 0; i < this.chunks.length; i++) {
      const v = this.chunks[i].vec;
      let s = 0;
      for (let j = 0; j < v.length; j++) s += v[j] * q[j];
      scored.push({ i, s });
    }
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, k);
  }

  private bm25(query: string, k: number) {
    const terms = [...new Set(tokenize(query))];
    if (!terms.length) return [];
    const N = this.chunks.length;
    const avg = this.totalLen / Math.max(1, N);
    const idf = new Map(terms.map((t) => {
      const df = this.df.get(t) ?? 0;
      return [t, Math.log(1 + (N - df + 0.5) / (df + 0.5))];
    }));
    const scored: { i: number; s: number }[] = [];
    for (let i = 0; i < N; i++) {
      const c = this.chunks[i];
      let s = 0;
      for (const t of terms) {
        const f = c.tf.get(t);
        if (!f) continue;
        s += idf.get(t)! * ((f * (K1 + 1)) / (f + K1 * (1 - B + (B * c.len) / avg)));
      }
      if (s > 0) scored.push({ i, s });
    }
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, k);
  }

  search(qvec: Float32Array, qtext: string, k = 40): Candidate[] {
    if (!this.chunks.length) return [];
    const vec = this.vectorSearch(qvec, 60);
    const bm = this.bm25(qtext, 60);
    const fused = new Map<number, Candidate>();
    const get = (i: number) => {
      let c = fused.get(i);
      if (!c) {
        const chunk = this.chunks[i];
        let cos = 0;
        for (let j = 0; j < chunk.vec.length; j++) cos += chunk.vec[j] * qvec[j];
        c = { chunk, rrf: 0, cosine: cos, vecRank: null, bm25Rank: null };
        fused.set(i, c);
      }
      return c;
    };
    vec.forEach((r, rank) => {
      const c = get(r.i);
      c.rrf += 1 / (RRF_K + rank + 1);
      c.vecRank = rank + 1;
    });
    bm.forEach((r, rank) => {
      const c = get(r.i);
      c.rrf += 1 / (RRF_K + rank + 1);
      c.bm25Rank = rank + 1;
    });
    return [...fused.values()].sort((a, b) => b.rrf - a.rrf).slice(0, k);
  }
}

export const index = new HybridIndex();

let lastSignature = '';
let lastCheck = 0;
/** documents currently in memory → their ingested_at, to reload only what changed */
const loadedDocs = new Map<string, number>();

async function signature() {
  const r = await get<{ n: number; m: number | null }>('SELECT COUNT(*) AS n, MAX(ingested_at) AS m FROM documents');
  return `${r?.n ?? 0}:${r?.m ?? 0}`;
}

type ChunkRow = { id: number; doc_id: string; page: number | null; text: string; embedding: Uint8Array; title: string; url: string | null; source_type: string };

/** Fetches the chunks of some documents, in groups of ~2,000 chunks queried in parallel (much faster over Turso's HTTP API). */
async function fetchChunks(docs: { id: string; chunks: number }[]): Promise<ChunkRow[]> {
  const groups: string[][] = [];
  let cur: string[] = [];
  let size = 0;
  for (const d of docs) {
    cur.push(d.id);
    size += d.chunks;
    if (size >= 2000) {
      groups.push(cur);
      cur = [];
      size = 0;
    }
  }
  if (cur.length) groups.push(cur);
  const out: ChunkRow[] = [];
  for (let i = 0; i < groups.length; i += 6) {
    const parts = await Promise.all(
      groups.slice(i, i + 6).map((ids) =>
        all<ChunkRow>(
          `SELECT c.id, c.doc_id, c.page, c.text, c.embedding, d.title, d.url, d.source_type
           FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.doc_id IN (${ids.map(() => '?').join(',')}) ORDER BY c.id`,
          ...ids,
        ),
      ),
    );
    for (const p of parts) out.push(...p);
  }
  return out;
}

function addRows(rows: ChunkRow[]) {
  for (const r of rows) {
    index.add({
      id: Number(r.id),
      docId: r.doc_id,
      title: r.title,
      page: r.page == null ? null : Number(r.page),
      url: r.url,
      sourceType: r.source_type,
      text: r.text,
      vec: fromBlob(r.embedding),
    });
  }
}

const listDocs = () => all<{ id: string; chunks: number; ingested_at: number }>('SELECT id, chunks, ingested_at FROM documents WHERE chunks > 0');

/** Loads all chunks from the database into memory (fetch first, then swap synchronously). */
export async function loadIndex() {
  const sig = await signature();
  const docs = await listDocs();
  const rows = await fetchChunks(docs);
  index.clear();
  loadedDocs.clear();
  addRows(rows);
  for (const d of docs) loadedDocs.set(d.id, Number(d.ingested_at));
  lastSignature = sig;
  lastCheck = Date.now();
  return index.size;
}

/**
 * Several serverless instances may run at once: every minute, check whether documents changed elsewhere
 * (new ingest, learned web page, removal) and apply only the difference — never a full reload.
 */
export async function ensureFreshIndex() {
  if (Date.now() - lastCheck < 60_000) return;
  lastCheck = Date.now();
  const sig = await signature();
  if (sig === lastSignature) return;
  const docs = await listDocs();
  const current = new Map(docs.map((d) => [d.id, Number(d.ingested_at)]));
  const changed = docs.filter((d) => loadedDocs.get(d.id) !== Number(d.ingested_at));
  const rows = await fetchChunks(changed);
  for (const id of loadedDocs.keys()) if (!current.has(id)) index.removeDoc(id);
  for (const d of changed) index.removeDoc(d.id);
  addRows(rows);
  loadedDocs.clear();
  for (const [id, at] of current) loadedDocs.set(id, at);
  lastSignature = sig;
}

/** Called after this instance changed the documents itself (its index is already up to date). */
export async function markIndexSynced() {
  const [sig, docs] = await Promise.all([signature(), listDocs()]);
  lastSignature = sig;
  loadedDocs.clear();
  for (const d of docs) loadedDocs.set(d.id, Number(d.ingested_at));
}
