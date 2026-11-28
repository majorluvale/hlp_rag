/**
 * Copies the locally indexed knowledge base (data/hlp.db: documents + chunks with embeddings)
 * to the Turso database used online — no re-embedding needed.
 * Usage: set DATABASE_URL and DATABASE_AUTH_TOKEN, then `npm run push:turso`.
 */
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { config } from '../config';
import { batch, get, initDb, q } from '../db';
import { setSettings } from '../settings';

if (!config.db.url || config.db.url.startsWith('file:')) {
  console.error('DATABASE_URL (Turso, libsql://…) is required.');
  process.exit(1);
}

const local = new DatabaseSync(path.join(config.dataDir, 'hlp.db'), { readOnly: true });
await initDb();

type Doc = { id: string; path: string | null; url: string | null; title: string; source_type: string; hash: string | null; pages: number | null; chunks: number; ingested_at: number };
const docs = local.prepare('SELECT * FROM documents WHERE chunks > 0 ORDER BY title').all() as Doc[];
console.log(`[push] ${docs.length} documents to copy → ${config.db.url}`);

let copied = 0;
let skipped = 0;
for (const d of docs) {
  if (await get('SELECT id FROM documents WHERE id = ? AND chunks > 0', d.id)) {
    skipped++;
    continue;
  }
  const rows = local.prepare('SELECT page, text, embedding FROM chunks WHERE doc_id = ? ORDER BY id').all(d.id) as {
    page: number | null;
    text: string;
    embedding: Uint8Array;
  }[];
  await batch([
    q('DELETE FROM chunks WHERE doc_id = ?', d.id),
    q('DELETE FROM documents WHERE id = ?', d.id),
    q(
      'INSERT INTO documents (id, path, url, title, source_type, hash, pages, chunks, ingested_at) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)',
      d.id, d.path ? path.basename(d.path) : null, d.url, d.title, d.source_type, d.hash, d.pages, d.ingested_at,
    ),
  ]);
  for (let i = 0; i < rows.length; i += 100) {
    await batch(rows.slice(i, i + 100).map((r) => q('INSERT INTO chunks (doc_id, page, text, embedding) VALUES (?, ?, ?, ?)', d.id, r.page, r.text, r.embedding)));
  }
  await batch([q('UPDATE documents SET chunks = ? WHERE id = ?', rows.length, d.id)]);
  copied++;
  console.log(`[push] ${copied + skipped}/${docs.length}  ${d.title} (${rows.length} extracts)`);
}

const meta = local.prepare("SELECT value FROM settings WHERE key = 'index.embedding'").get() as { value: string } | undefined;
await setSettings({ 'index.embedding': meta?.value ?? 'multilingual-e5-small:384' }, 'push-to-turso');
const total = await get<{ n: number }>('SELECT COUNT(*) n FROM chunks');
console.log(`[push] done: ${copied} copied, ${skipped} already online · ${total?.n} extracts in Turso`);
process.exit(0);
