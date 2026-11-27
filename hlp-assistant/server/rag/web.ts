import crypto from 'node:crypto';
import * as cheerio from 'cheerio';
import { extractText, getDocumentProxy } from 'unpdf';
import { config } from '../config';
import { get, now, run } from '../db';
import { addDocument } from './ingest';
import { cleanText } from './text';

const UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36';
const CACHE_TTL = 7 * 24 * 3600 * 1000;
const MAX_BYTES = 8 * 1024 * 1024;

export interface WebResult {
  title: string;
  url: string;
  snippet: string;
}

export function isAllowedUrl(raw: string): boolean {
  try {
    const u = new URL(raw);
    if (u.protocol !== 'https:' && u.protocol !== 'http:') return false;
    const host = u.hostname.toLowerCase();
    return config.web.domains.some((d) => host === d || host.endsWith('.' + d));
  } catch {
    return false;
  }
}

function resolveDomain(site?: string): string[] {
  if (!site) return config.web.domains;
  const s = site.toLowerCase().replace(/^https?:\/\//, '').replace(/\/.*$/, '');
  const match = config.web.domains.filter((d) => s === d || s.endsWith('.' + d) || d.includes(s));
  return match.length ? match : config.web.domains;
}

async function tavilySearch(query: string, domains: string[]): Promise<WebResult[]> {
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${config.web.tavilyKey}` },
    body: JSON.stringify({ query, include_domains: domains, max_results: 6, search_depth: 'advanced' }),
    signal: AbortSignal.timeout(20000),
  });
  if (!res.ok) throw new Error(`Tavily ${res.status}`);
  const data = (await res.json()) as { results?: { title: string; url: string; content: string }[] };
  return (data.results ?? []).map((r) => ({ title: r.title, url: r.url, snippet: r.content?.slice(0, 500) ?? '' }));
}

async function duckDuckGoSearch(query: string, domains: string[]): Promise<WebResult[]> {
  const sites = domains.map((d) => `site:${d}`).join(' OR ');
  const run1 = async (q: string) => {
    const res = await fetch('https://html.duckduckgo.com/html/', {
      method: 'POST',
      headers: { 'User-Agent': UA, 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({ q }).toString(),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) throw new Error(`DuckDuckGo ${res.status}`);
    const $ = cheerio.load(await res.text());
    const out: WebResult[] = [];
    $('.result').each((_, el) => {
      const a = $(el).find('a.result__a');
      let href = a.attr('href') ?? '';
      if (href.includes('uddg=')) {
        href = new URL(href, 'https://duckduckgo.com').searchParams.get('uddg') ?? href;
      }
      if (!href.startsWith('http')) return;
      out.push({ title: a.text().trim(), url: href, snippet: $(el).find('.result__snippet').text().trim() });
    });
    return out.filter((r) => isAllowedUrl(r.url));
  };
  let results = await run1(`${query} ${domains.length === 1 ? `site:${domains[0]}` : `(${sites})`}`);
  if (!results.length && domains.length > 1) {
    // fall back: one domain at a time for the most relevant sites
    for (const d of domains.slice(0, 3)) {
      results.push(...(await run1(`${query} site:${d}`)));
      if (results.length >= 4) break;
    }
  }
  return results.slice(0, 6);
}

/** Searches only the trusted humanitarian domains (HLP AoR/GPC, NRC, UN-Habitat, CCCM, Shelter/SLSC…). */
export async function searchWeb(query: string, site?: string): Promise<{ provider: string; results: WebResult[] }> {
  const domains = resolveDomain(site);
  if (config.web.tavilyKey) {
    try {
      return { provider: 'tavily', results: await tavilySearch(query, domains) };
    } catch (e) {
      console.warn('[web] Tavily indisponible, repli DuckDuckGo :', (e as Error).message);
    }
  }
  let results = await duckDuckGoSearch(query, domains);
  if (!results.length && domains.length < config.web.domains.length) results = await duckDuckGoSearch(query, config.web.domains);
  return { provider: 'duckduckgo', results };
}

export interface WebPage {
  url: string;
  title: string;
  text: string;
  cached: boolean;
}

export async function fetchPage(url: string): Promise<WebPage> {
  if (!isAllowedUrl(url)) throw new Error(`Domaine non autorisé : ${url}`);
  const cached = get<{ title: string; text: string; fetched_at: number }>('SELECT * FROM web_cache WHERE url = ?', url);
  if (cached && now() - cached.fetched_at < CACHE_TTL) return { url, title: cached.title, text: cached.text, cached: true };

  const res = await fetch(url, { headers: { 'User-Agent': UA, Accept: 'text/html,application/pdf;q=0.9,*/*;q=0.5' }, signal: AbortSignal.timeout(20000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  if (!isAllowedUrl(res.url)) throw new Error(`Redirection vers un domaine non autorisé : ${res.url}`);
  const len = Number(res.headers.get('content-length') ?? 0);
  if (len > MAX_BYTES) throw new Error('Document trop volumineux');
  const type = res.headers.get('content-type') ?? '';
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.byteLength > MAX_BYTES) throw new Error('Document trop volumineux');

  let title = url;
  let text = '';
  if (type.includes('pdf') || /\.pdf($|\?)/i.test(url)) {
    const pdf = await getDocumentProxy(new Uint8Array(buf));
    const out = await extractText(pdf, { mergePages: true });
    text = cleanText(out.text as string);
    title = decodeURIComponent(url.split('/').pop() ?? url);
  } else {
    const $ = cheerio.load(buf.toString('utf8'));
    $('script,style,noscript,nav,footer,header,form,svg,iframe,aside,.cookie,.menu').remove();
    title = ($('meta[property="og:title"]').attr('content') || $('title').text() || $('h1').first().text() || url).trim();
    const main = $('main').text() || $('article').text() || $('[role=main]').text() || $('body').text();
    text = cleanText(main);
  }
  text = text.slice(0, 40000);
  run(
    'INSERT INTO web_cache (url, title, text, fetched_at) VALUES (?, ?, ?, ?) ON CONFLICT(url) DO UPDATE SET title = excluded.title, text = excluded.text, fetched_at = excluded.fetched_at',
    url, title, text, now(),
  );
  return { url, title, text, cached: false };
}

/** The assistant "learns" from the official pages it reads: they join the local knowledge base. */
export function learnFromPage(page: WebPage) {
  if (!config.web.learnFromWeb || page.text.length < 600) return;
  const id = 'web_' + crypto.createHash('sha1').update(page.url).digest('hex').slice(0, 16);
  if (get('SELECT id FROM documents WHERE id = ?', id)) return;
  addDocument({
    id,
    title: page.title.slice(0, 200),
    sourceType: 'web',
    url: page.url,
    pages: [{ page: null, text: page.text }],
  })
    .then((n) => console.log(`[web] page apprise (${n} chunks) : ${page.url}`))
    .catch((e) => console.warn('[web] apprentissage impossible :', (e as Error).message));
}
