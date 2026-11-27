import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { extractText, getDocumentProxy } from 'unpdf';
import { config } from '../config';
import { get, now, run, toBlob, transaction } from '../db';
import { embed } from './embeddings';
import { index } from './hybridIndex';
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

const SUPPORTED = /\.(pdf|txt|md)$/i;

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
  return [cleanText(buf.toString('utf8'))];
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
    const batch = pieces.slice(i, i + 16);
    vectors.push(...(await embed(batch.map((b) => `${doc.title}. ${b.text}`), 'passage')));
  }

  const ids: number[] = [];
  transaction(() => {
    run('DELETE FROM documents WHERE id = ?', doc.id);
    run(
      `INSERT INTO documents (id, path, url, title, source_type, hash, pages, chunks, ingested_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      doc.id, doc.path, doc.url, doc.title, doc.sourceType, doc.hash, doc.pages.length, pieces.length, now(),
    );
    pieces.forEach((p, i) => {
      const r = run('INSERT INTO chunks (doc_id, page, text, embedding) VALUES (?, ?, ?, ?)', doc.id, p.page, p.text, toBlob(vectors[i]));
      ids.push(Number(r.lastInsertRowid));
    });
  });

  index.removeDoc(doc.id);
  pieces.forEach((p, i) =>
    index.add({
      id: ids[i],
      docId: doc.id,
      title: doc.title,
      page: p.page,
      url: doc.url ?? null,
      sourceType: doc.sourceType,
      text: p.text,
      vec: vectors[i],
    }),
  );
  return pieces.length;
}

export async function ingestAll(): Promise<IngestStatus> {
  if (ingestStatus.running) return ingestStatus;
  const files = listFiles();
  Object.assign(ingestStatus, { running: true, total: files.length, done: 0, added: 0, skipped: 0, errors: [], finishedAt: null });
  try {
    for (const file of files) {
      ingestStatus.current = path.basename(file);
      try {
        const buf = fs.readFileSync(file);
        const hash = crypto.createHash('sha1').update(buf).digest('hex');
        const id = 'doc_' + hash.slice(0, 16);
        const existing = get<{ hash: string }>('SELECT hash FROM documents WHERE id = ?', id);
        if (existing) {
          ingestStatus.skipped++;
        } else {
          const pages = await extractPages(file, buf);
          const n = await addDocument({
            id,
            title: prettyTitle(path.basename(file)),
            sourceType: 'doc',
            path: file,
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

// CLI: npm run ingest
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  console.log('[ingest] dossiers :', config.docsDirs.join(' | '));
  ingestAll().then((s) => {
    console.log(`[ingest] terminé : ${s.added} ajoutés, ${s.skipped} déjà indexés, ${s.errors.length} erreurs`);
    process.exit(0);
  });
}
