import path from 'node:path';
import OpenAI from 'openai';
import { config } from '../config';

/**
 * Three interchangeable embedding providers:
 *  - local       : transformers.js on CPU (free, private) — default
 *  - huggingface : HF Inference API, same multilingual-e5-small model → an index built locally stays valid online (Vercel)
 *  - openai      : any OpenAI-compatible /embeddings endpoint (requires re-indexing)
 */
const provider = config.embedding.provider;
const model =
  config.embedding.model ||
  (provider === 'local' ? 'Xenova/multilingual-e5-small' : provider === 'huggingface' ? 'intfloat/multilingual-e5-small' : 'text-embedding-3-small');

/** Vectors are comparable across providers when the model family matches (Xenova/… is the ONNX port of intfloat/…). */
export const embeddingInfo = { provider, model, family: model.split('/').pop()!.toLowerCase() };
const isE5 = /e5/i.test(model);

function normalise(v: ArrayLike<number>): Float32Array {
  const out = Float32Array.from(v);
  let n = 0;
  for (let i = 0; i < out.length; i++) n += out[i] * out[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < out.length; i++) out[i] /= n;
  return out;
}

// ---- local (transformers.js) — imported lazily and never bundled for serverless ----
let extractor: Promise<any> | null = null;
async function localEmbed(inputs: string[]): Promise<Float32Array[]> {
  if (!extractor) {
    console.log(`[embeddings] chargement du modèle local ${model}…`);
    const spec = ['@huggingface', 'transformers'].join('/');
    extractor = import(spec).then((m: any) => {
      m.env.cacheDir = path.join(config.dataDir, 'models');
      return m.pipeline('feature-extraction', model, { dtype: 'q8' });
    });
  }
  const ex = await extractor;
  const out = await ex(inputs, { pooling: 'mean', normalize: true });
  const [n, d] = out.dims as [number, number];
  const data = out.data as Float32Array;
  return Array.from({ length: n }, (_, i) => data.slice(i * d, (i + 1) * d));
}

// ---- Hugging Face Inference API ----
async function hfEmbed(inputs: string[]): Promise<Float32Array[]> {
  if (!config.embedding.hfToken) throw new Error('HF_TOKEN manquant pour EMBEDDING_PROVIDER=huggingface');
  const url = `https://router.huggingface.co/hf-inference/models/${model}/pipeline/feature-extraction`;
  let lastErr = '';
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { Authorization: `Bearer ${config.embedding.hfToken}`, 'Content-Type': 'application/json', 'x-wait-for-model': 'true' },
      body: JSON.stringify({ inputs, normalize: true }),
      signal: AbortSignal.timeout(45_000),
    });
    if (res.ok) {
      const data = (await res.json()) as unknown[];
      return data.map((v) => {
        // sentence vector (number[]) or token vectors (number[][]) → mean pooling
        if (Array.isArray(v) && Array.isArray(v[0])) {
          const toks = v as number[][];
          const mean = new Array(toks[0].length).fill(0);
          for (const t of toks) for (let i = 0; i < t.length; i++) mean[i] += t[i] / toks.length;
          return normalise(mean);
        }
        return normalise(v as number[]);
      });
    }
    lastErr = `${res.status} ${(await res.text()).slice(0, 200)}`;
    if (res.status !== 503 && res.status !== 429) break;
    await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
  }
  throw new Error(`Hugging Face embeddings : ${lastErr}`);
}

// ---- OpenAI-compatible ----
let oa: OpenAI | null = null;
async function openaiEmbed(inputs: string[]): Promise<Float32Array[]> {
  oa ??= new OpenAI({ baseURL: config.embedding.baseURL, apiKey: config.embedding.apiKey || 'missing-key' });
  const r = await oa.embeddings.create({ model, input: inputs });
  return r.data.sort((a, b) => a.index - b.index).map((d) => normalise(d.embedding));
}

/** Embeds texts. e5 models expect "query: " / "passage: " prefixes. Vectors are L2-normalised. */
export async function embed(texts: string[], kind: 'query' | 'passage'): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  const inputs = texts.map((t) => (isE5 ? `${kind}: ${t}` : t));
  if (provider === 'huggingface') return hfEmbed(inputs);
  if (provider === 'openai') return openaiEmbed(inputs);
  return localEmbed(inputs);
}

export async function embedOne(text: string, kind: 'query' | 'passage'): Promise<Float32Array> {
  return (await embed([text], kind))[0];
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}
