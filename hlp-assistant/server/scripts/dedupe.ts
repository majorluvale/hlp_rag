/**
 * Finds documents whose text is (almost) entirely contained in another document — same report saved twice,
 * "_0"/"_1" copies, a newsletter both as a web page and as a PDF… — and removes the extra copies.
 *
 * Usage:  npm run dedupe            → lists the duplicates (nothing is changed)
 *         npm run dedupe -- --apply → removes them from the knowledge base and moves their files to
 *                                     <folder>/_duplicates (ignored by the ingest, easy to restore)
 * Works on the local database, or on Turso when DATABASE_URL is set.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { config } from '../config';
import { all, batch, initDb, q } from '../db';
import { markIndexSynced } from '../rag/hybridIndex';
import { normalize } from '../rag/text';

const apply = process.argv.includes('--apply');
// share of a document's text found in a larger one (override with --threshold=0.7)
const THRESHOLD = Number(process.argv.find((a) => a.startsWith('--threshold='))?.split('=')[1] ?? 0.85);

await initDb();
type Doc = { id: string; title: string; path: string | null; url: string | null; chunks: number };
const docs = await all<Doc>('SELECT id, title, path, url, chunks FROM documents WHERE chunks > 0');

// fingerprint = hashes of 8-word shingles (1 in 4 kept, the same ones for every document)
const prints = new Map<string, Set<number>>();
for (const d of docs) {
  const rows = await all<{ text: string }>('SELECT text FROM chunks WHERE doc_id = ? ORDER BY id', d.id);
  const words = normalize(rows.map((r) => r.text).join(' ')).split(/[^a-z0-9]+/).filter(Boolean);
  const set = new Set<number>();
  for (let i = 0; i + 8 <= words.length; i++) {
    const h = crypto.createHash('md5').update(words.slice(i, i + 8).join(' ')).digest().readUInt32LE(0);
    if (h % 4 === 0) set.add(h);
  }
  prints.set(d.id, set);
}

// the copy we keep: the one with a source link, then the most complete one
const rank = (d: Doc) => (d.url ? 1e9 : 0) + prints.get(d.id)!.size;
const sorted = [...docs].sort((a, b) => rank(b) - rank(a));
const kept: Doc[] = [];
const dupes: { doc: Doc; of: Doc; share: number }[] = [];
for (const d of sorted) {
  const mine = prints.get(d.id)!;
  if (mine.size < 3) {
    kept.push(d);
    continue;
  }
  let match: { of: Doc; share: number } | null = null;
  for (const k of kept) {
    const theirs = prints.get(k.id)!;
    let common = 0;
    for (const h of mine) if (theirs.has(h)) common++;
    const share = common / mine.size;
    if (share >= THRESHOLD && (!match || share > match.share)) match = { of: k, share };
  }
  if (match) dupes.push({ doc: d, ...match });
  else kept.push(d);
}

for (const x of dupes) console.log(`✗ ${x.doc.title}  (${x.doc.chunks} extracts)\n    ${(x.share * 100).toFixed(0)}% contained in: ${x.of.title}`);
const extra = dupes.reduce((s, x) => s + x.doc.chunks, 0);
console.log(`\n${dupes.length} duplicate document(s) out of ${docs.length} — ${extra} extracts`);
if (!apply) {
  console.log('Nothing changed. Run again with --apply to remove them.');
  process.exit(0);
}

for (const { doc } of dupes) {
  await batch([q('DELETE FROM chunks WHERE doc_id = ?', doc.id), q('DELETE FROM documents WHERE id = ?', doc.id)]);
  // move the file aside so the next ingest does not add it back
  if (!doc.path) continue;
  for (const dir of config.docsDirs) {
    const found = findFile(dir, path.basename(doc.path));
    if (!found) continue;
    const dest = path.join(path.dirname(found), '_duplicates');
    fs.mkdirSync(dest, { recursive: true });
    fs.renameSync(found, path.join(dest, path.basename(found)));
  }
}
await markIndexSynced();
console.log(`Removed ${dupes.length} documents (${extra} extracts). Their files are in the _duplicates folders.`);
process.exit(0);

function findFile(dir: string, name: string): string | null {
  if (!fs.existsSync(dir)) return null;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory() && !e.name.startsWith('_')) {
      const f = findFile(p, name);
      if (f) return f;
    } else if (e.name === name) return p;
  }
  return null;
}
