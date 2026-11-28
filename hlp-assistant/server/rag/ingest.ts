import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { extractText, getDocumentProxy } from 'unpdf';
import { config } from '../config';
import { all, batch, get, initDb, now, q, run, toBlob } from '../db';
import { setSettings } from '../settings';
import { embed, embeddingInfo } from './embeddings';
import { index, markIndexSynced } from './hybridIndex';
import { chunkText, cleanText, prettyTitle } from './text';

export interface IngestStatus {
  running: boolean;
  total: number;
  done: number;
  current: string | null;
  added: number;
  skipped: number;
  errors: string[];
  finishedAt: number | null;
}

export const ingestStatus: IngestStatus = {
  running: false,
  total: 0,
  done: 0,
  current: null,
  added: 0,
  skipped: 0,
  errors: [],
  finishedAt: null,
};

const SUPPORTED = /\.(pdf|txt|md|docx)$/i;

function listFiles(): string[] {
  const files: string[] = [];
  const walk = (dir: string) => {
    if (!fs.existsSync(dir)) return;
    for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) walk(p);
      else if (SUPPORTED.test(e.name)) files.push(p);
    }
  };
  config.docsDirs.forEach(walk);
  return files;
}

async function extractPages(file: string, buf: Buffer): Promise<string[]> {
  if (/\.pdf$/i.test(file)) {
    const pdf = await getDocumentProxy(new Uint8Array(buf));
    const { text } = await extractText(pdf, { mergePages: false });
    return (text as string[]).map(cleanText);
  }
  if (/\.docx$/i.test(file)) {
    const mammoth = await import('mammoth');
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return [cleanText(value)];
  }
  return [cleanText(buf.toString('utf8'))];
}

/** Original URL of a file, from the `_sources.json` written next to it by `npm run import:links`. */
const sourceMaps = new Map<string, Record<string, string>>();
function sourceUrl(file: string): string | null {
  const dir = path.dirname(file);
  if (!sourceMaps.has(dir)) {
    const f = path.join(dir, '_sources.json');
    try {
      sourceMaps.set(dir, fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : {});
    } catch {
      sourceMaps.set(dir, {});
    }
  }
  return sourceMaps.get(dir)![path.basename(file)] ?? null;
}

/** Markdown files written by the importer start with "# Title": use it instead of the file name. */
function fileTitle(file: string, buf: Buffer): string {
  if (/\.md$/i.test(file)) {
    const m = buf.toString('utf8', 0, 600).match(/^# (.+)$/m);
    if (m) return m[1].trim().slice(0, 200);
  }
  return prettyTitle(path.basename(file));
}

/** Stores a document's chunks (with embeddings) and adds them to the live index. */
export async function addDocument(doc: {
  id: string;
  title: string;
  sourceType: 'doc' | 'web';
  path?: string | null;
  url?: string | null;
  hash?: string | null;
  pages: { page: number | null; text: string }[];
}) {
  const pieces: { page: number | null; text: string }[] = [];
  for (const p of doc.pages) for (const t of chunkText(p.text)) pieces.push({ page: p.page, text: t });
  if (!pieces.length) return 0;

  const vectors: Float32Array[] = [];
  for (let i = 0; i < pieces.length; i += 16) {
    const slice = pieces.slice(i, i + 16);
    vectors.push(...(await embed(slice.map((b) => `${doc.title}. ${b.text}`), 'passage')));
  }

  // The document row is written first with chunks = 0 and only marked complete at the end,
  // so an interrupted ingest is retried next time instead of leaving a half-indexed document.
  await batch([
    q('DELETE FROM chunks WHERE doc_id = ?', doc.id),
    q('DELETE FROM documents WHERE id = ?', doc.id),
    q(
      `INSERT INTO documents (id, path, url, title, source_type, hash, pages, chunks, ingested_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)`,
      doc.id, doc.path, doc.url, doc.title, doc.sourceType, doc.hash, doc.pages.length, now(),
    ),
  ]);
  for (let i = 0; i < pieces.length; i += 50) {
    await batch(
      pieces.slice(i, i + 50).map((p, j) =>
        q('INSERT INTO chunks (doc_id, page, text, embedding) VALUES (?, ?, ?, ?)', doc.id, p.page, p.text, toBlob(vectors[i + j])),
      ),
    );
  }
  await run('UPDATE documents SET chunks = ?, ingested_at = ? WHERE id = ?', pieces.length, now(), doc.id);
  const ids = await all<{ id: number }>('SELECT id FROM chunks WHERE doc_id = ? ORDER BY id', doc.id);

  index.removeDoc(doc.id);
  pieces.forEach((p, i) =>
    index.add({
      id: Number(ids[i]?.id ?? 0),
      docId: doc.id,
      title: doc.title,
      page: p.page,
      url: doc.url ?? null,
      sourceType: doc.sourceType,
      text: p.text,
      vec: vectors[i],
    }),
  );
  await markIndexSynced();
  await setSettings({ 'index.embedding': `${embeddingInfo.family}:${vectors[0].length}` }, 'ingest');
  return pieces.length;
}

export async function ingestAll(): Promise<IngestStatus> {
  if (ingestStatus.running) return ingestStatus;
  sourceMaps.clear();
  const files = listFiles();
  Object.assign(ingestStatus, { running: true, total: files.length, done: 0, added: 0, skipped: 0, errors: [], finishedAt: null });
  try {
    for (const file of files) {
      ingestStatus.current = path.basename(file);
      try {
        const buf = fs.readFileSync(file);
        const hash = crypto.createHash('sha1').update(buf).digest('hex');
        const id = 'doc_' + hash.slice(0, 16);
        if (await get('SELECT id FROM documents WHERE id = ? AND chunks > 0', id)) {
          ingestStatus.skipped++;
        } else {
          const pages = await extractPages(file, buf);
          const n = await addDocument({
            id,
            title: fileTitle(file, buf),
            sourceType: 'doc',
            path: path.basename(file),
            url: sourceUrl(file),
            hash,
            pages: pages.map((text, i) => ({ page: pages.length > 1 ? i + 1 : null, text })),
          });
          ingestStatus.added++;
          console.log(`[ingest] ${path.basename(file)} → ${n} chunks`);
        }
      } catch (e) {
        const msg = `${path.basename(file)}: ${(e as Error).message}`;
        ingestStatus.errors.push(msg);
        console.warn('[ingest] erreur', msg);
      }
      ingestStatus.done++;
    }
  } finally {
    ingestStatus.running = false;
    ingestStatus.current = null;
    ingestStatus.finishedAt = Date.now();
  }
  return ingestStatus;
}

// CLI: npm run ingest  (works against local SQLite or, with DATABASE_URL set, against Turso)
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  (async () => {
    await initDb();
    console.log(`[ingest] base : ${config.db.url ? 'Turso ' + config.db.url : 'SQLite local'} · embeddings : ${embeddingInfo.provider} ${embeddingInfo.model}`);
    console.log('[ingest] dossiers :', config.docsDirs.join(' | '));
    const s = await ingestAll();
    console.log(`[ingest] terminé : ${s.added} ajoutés, ${s.skipped} déjà indexés, ${s.errors.length} erreurs`);
    process.exit(0);
  })().catch((e) => {
    console.error('[ingest]', e);
    process.exit(1);
  });
}
