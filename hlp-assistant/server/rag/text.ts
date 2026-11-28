const STOPWORDS = new Set(
  (
    'le la les un une des du de d l et ou en au aux ce ces cet cette pour par sur dans avec sans est sont etre ' +
    'a qui que quoi dont ou il elle ils elles on nous vous je tu se sa son ses leur leurs plus moins pas ne ' +
    'comment quel quelle quels quelles quand est-ce y the of and or to in on for with by is are be as at an ' +
    'this that these those it its from what how which who when why do does can should would will about into'
  ).split(/\s+/),
);

export function normalize(s: string): string {
  return s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
}

export function tokenize(s: string): string[] {
  return normalize(s)
    .split(/[^a-z0-9]+/)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t))
    .map((t) => (t.length > 4 && t.endsWith('s') ? t.slice(0, -1) : t));
}

export function cleanText(s: string): string {
  return s
    .replace(/\u0000/g, '')
    .replace(/(\w)-\n(\w)/g, '$1$2')
    .replace(/[ \t\f\v]+/g, ' ')
    .replace(/ *\n */g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

/** Sentence-aware chunking with overlap. */
export function chunkText(text: string, size = 1100, overlap = 200): string[] {
  const flat = text.replace(/\s*\n\s*/g, ' ').trim();
  if (!flat) return [];
  const sentences = flat.split(/(?<=[.!?;:])\s+/);
  const chunks: string[] = [];
  let cur = '';
  const push = () => {
    const c = cur.trim();
    if (c.length > 40) chunks.push(c);
  };
  for (let s of sentences) {
    while (s.length > size) {
      // very long "sentence" (tables, lists): hard split
      const head = s.slice(0, size);
      s = s.slice(size - overlap);
      if (cur) {
        push();
        cur = '';
      }
      chunks.push(head);
    }
    if (cur.length + s.length + 1 > size && cur) {
      push();
      cur = cur.slice(Math.max(0, cur.length - overlap));
      const sp = cur.indexOf(' ');
      if (sp > 0) cur = cur.slice(sp + 1);
    }
    cur += (cur ? ' ' : '') + s;
  }
  push();
  return chunks;
}

export function prettyTitle(fileName: string): string {
  return fileName
    .replace(/\.[a-z0-9]+$/i, '')
    .replace(/[_]+/g, ' ')
    .replace(/^\d+[._\s-]*/, '')
    .replace(/\s+/g, ' ')
    .trim();
}
