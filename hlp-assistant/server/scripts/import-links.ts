/**
 * Downloads a list of knowledge sources into a documents folder, ready for `npm run ingest`:
 *   - PDF / DOCX files       → saved as-is
 *   - web pages, newsletters → main text saved as Markdown
 *   - YouTube videos         → title, description and (when available) the transcript, as Markdown
 *   - Google Docs            → exported as DOCX (the document must be shared "anyone with the link")
 * Each file's original URL is recorded in `_sources.json`, so answers can link back to the source.
 *
 * Usage: npm run import:links [-- links.txt [outDir]]
 *   Lines starting with /sites are taken as globalprotectioncluster.org paths. Lines starting with # are ignored.
 */
import fs from 'node:fs';
import path from 'node:path';
import * as cheerio from 'cheerio';

const listFile = process.argv[2] ?? 'sources/links.txt';
const outDir = path.resolve(process.argv[3] ?? '../../docs/imported');
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36';
const existingDirs = ['../../docs', '../../knowledge', 'data/docs'].map((d) => path.resolve(d));

fs.mkdirSync(outDir, { recursive: true });
const sourcesFile = path.join(outDir, '_sources.json');
const sources: Record<string, string> = fs.existsSync(sourcesFile) ? JSON.parse(fs.readFileSync(sourcesFile, 'utf8')) : {};
const save = () => fs.writeFileSync(sourcesFile, JSON.stringify(sources, null, 2));

const links = [
  ...new Set(
    fs
      .readFileSync(listFile, 'utf8')
      .split(/\r?\n/)
      .map((l) => l.trim().replace(/^"|"$/g, '').trim())
      .filter((l) => l && !l.startsWith('#'))
      .map((l) => (l.startsWith('/sites/') ? 'https://www.globalprotectioncluster.org' + l : l)),
  ),
];

const safeName = (s: string) =>
  s
    .replace(/\[\d+\]/g, '')
    .replace(/[<>:"/\\|?*\u0000-\u001f]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 150);

const alreadyHave = (name: string) => existingDirs.some((d) => fs.existsSync(path.join(d, name)));

async function get(url: string) {
  const res = await fetch(url, { headers: { 'User-Agent': UA, 'Accept-Language': 'en,fr;q=0.8' }, redirect: 'follow', signal: AbortSignal.timeout(90_000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res;
}

function writeMd(name: string, url: string, title: string, body: string) {
  const file = safeName(name) + '.md';
  fs.writeFileSync(path.join(outDir, file), `# ${title.trim()}\n\nSource: ${url}\n\n${body.trim()}\n`);
  sources[file] = url;
  return file;
}

const decodeEntities = (s: string) =>
  s.replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)));

async function youtube(url: string) {
  const id = new URL(url).searchParams.get('v')!;
  const html = await (await get(`https://www.youtube.com/watch?v=${id}&hl=en`)).text();
  const m = html.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\})\s*;\s*(?:var\s|<\/script>)/s);
  const player = m ? JSON.parse(m[1]) : {};
  const title: string = player.videoDetails?.title ?? id;
  const author: string = player.videoDetails?.author ?? '';
  const description: string = player.videoDetails?.shortDescription ?? '';
  let transcript = '';
  const tracks: { baseUrl: string; languageCode: string; kind?: string }[] = player.captions?.playerCaptionsTracklistRenderer?.captionTracks ?? [];
  const track = tracks.find((t) => t.languageCode.startsWith('en') && t.kind !== 'asr') ?? tracks.find((t) => t.languageCode.startsWith('en')) ?? tracks[0];
  if (track) {
    try {
      const xml = await (await get(track.baseUrl)).text();
      transcript = [...xml.matchAll(/<text[^>]*>([\s\S]*?)<\/text>/g)].map((x) => decodeEntities(decodeEntities(x[1])).replace(/\n/g, ' ')).join(' ');
    } catch {
      /* captions not reachable: keep title + description */
    }
  }
  if (!description && !transcript) throw new Error('no description or transcript');
  const body = [author && `Channel: ${author}`, description && `## Description\n\n${description}`, transcript && `## Transcript\n\n${transcript}`].filter(Boolean).join('\n\n');
  return { file: writeMd(`YouTube - ${title}`, url, `${title} (video)`, body), note: transcript ? 'transcript' : 'description only' };
}

async function googleDoc(url: string) {
  const id = url.match(/\/document\/d\/([^/]+)/)![1];
  const res = await get(`https://docs.google.com/document/d/${id}/export?format=docx`);
  const type = res.headers.get('content-type') ?? '';
  if (type.includes('text/html')) throw new Error('not shared publicly (sign-in page returned)');
  const cd = res.headers.get('content-disposition') ?? '';
  const named = decodeURIComponent(cd.match(/filename\*=UTF-8''([^;]+)/)?.[1] ?? cd.match(/filename="([^"]+)"/)?.[1] ?? `google-doc-${id}.docx`);
  const file = safeName(named.replace(/\.docx$/i, '')) + '.docx';
  fs.writeFileSync(path.join(outDir, file), Buffer.from(await res.arrayBuffer()));
  sources[file] = url;
  return { file, note: '' };
}

async function webPage(url: string) {
  const res = await get(url);
  const type = res.headers.get('content-type') ?? '';
  if (type.includes('pdf')) return binary(url, res, '.pdf');
  const html = await res.text();
  let $ = cheerio.load(html);
  $('script,style,noscript,nav,footer,header,form,svg,iframe,aside,button,.cookie,.menu').remove();
  // some templates wrap the whole content in <header>/<form>: fall back to a lighter clean-up
  if ($('body').text().trim().length < 300) {
    $ = cheerio.load(html);
    $('script,style,noscript,svg,iframe').remove();
  }
  const title = ($('meta[property="og:title"]').attr('content') || $('title').text() || $('h1').first().text() || url).trim();
  $('br').replaceWith('\n');
  $('p,li,h1,h2,h3,h4,tr,div').each((_, el) => void $(el).append('\n'));
  const root = $('main').length ? $('main') : $('article').length ? $('article') : $('body');
  const text = root.text().replace(/[ \t]+/g, ' ').replace(/ *\n */g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  if (text.length < 300) throw new Error(`page too short (${text.length} chars)`);
  // the URL slug keeps file names unique (newsletters often share the same title)
  const slug = new URL(url).pathname.split('/').filter(Boolean).pop() ?? '';
  return { file: writeMd(`${title} - ${slug}`, url, title, text), note: `${(text.length / 1000).toFixed(0)}k chars` };
}

async function binary(url: string, res: Response, ext: string) {
  let name = decodeURIComponent(new URL(url).pathname.split('/').pop() || 'document');
  name = safeName(name.replace(/\.(pdf|docx)$/i, '')) + ext;
  if (alreadyHave(name) && !fs.existsSync(path.join(outDir, name))) return { file: name, note: 'already in the documents folder' };
  const buf = Buffer.from(await res.arrayBuffer());
  if (ext === '.pdf' && buf.subarray(0, 5).toString() !== '%PDF-') throw new Error('not a PDF');
  fs.writeFileSync(path.join(outDir, name), buf);
  sources[name] = url;
  return { file: name, note: `${(buf.byteLength / 1024 / 1024).toFixed(1)} MB` };
}

async function one(url: string) {
  const host = new URL(url).hostname;
  if (/youtube\.com|youtu\.be/.test(host)) return youtube(url);
  if (host === 'docs.google.com') return googleDoc(url);
  const ext = url.match(/\.(pdf|docx)(\?|$)/i)?.[1]?.toLowerCase();
  if (ext) {
    const name = safeName(decodeURIComponent(new URL(url).pathname.split('/').pop()!).replace(/\.(pdf|docx)$/i, '')) + '.' + ext;
    if (fs.existsSync(path.join(outDir, name))) return { file: name, note: 'already downloaded' };
    if (alreadyHave(name)) return { file: name, note: 'already in the documents folder' };
    return binary(url, await get(url), '.' + ext);
  }
  return webPage(url);
}

let ok = 0;
const failed: string[] = [];
const queue = [...links];
const workers = Array.from({ length: 4 }, async () => {
  while (queue.length) {
    const url = queue.shift()!;
    try {
      const r = await one(url);
      ok++;
      console.log(`✓ ${r.file}${r.note ? '  (' + r.note + ')' : ''}`);
      save();
    } catch (e) {
      failed.push(`${url}  → ${(e as Error).message}`);
      console.log(`✗ ${url}  → ${(e as Error).message}`);
    }
  }
});
await Promise.all(workers);
save();
console.log(`\n${ok}/${links.length} sources ready in ${outDir}`);
if (failed.length) console.log(`Not imported (${failed.length}):\n  ` + failed.join('\n  '));
