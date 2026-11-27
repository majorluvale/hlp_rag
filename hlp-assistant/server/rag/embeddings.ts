import path from 'node:path';
import { pipeline, env, type FeatureExtractionPipeline } from '@huggingface/transformers';
import { config } from '../config';

env.cacheDir = path.join(config.dataDir, 'models');

let extractor: Promise<FeatureExtractionPipeline> | null = null;
const isE5 = /e5/i.test(config.embeddingModel);

function load(): Promise<FeatureExtractionPipeline> {
  if (!extractor) {
    console.log(`[embeddings] chargement du modèle ${config.embeddingModel}…`);
    extractor = pipeline('feature-extraction', config.embeddingModel, { dtype: 'q8' }) as Promise<FeatureExtractionPipeline>;
  }
  return extractor;
}

/** Embeds texts. e5 models expect "query: " / "passage: " prefixes. Vectors are L2-normalised. */
export async function embed(texts: string[], kind: 'query' | 'passage'): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  const ex = await load();
  const inputs = texts.map((t) => (isE5 ? `${kind}: ${t}` : t));
  const out = await ex(inputs, { pooling: 'mean', normalize: true });
  const [n, d] = out.dims as [number, number];
  const data = out.data as Float32Array;
  return Array.from({ length: n }, (_, i) => data.slice(i * d, (i + 1) * d));
}

export async function embedOne(text: string, kind: 'query' | 'passage'): Promise<Float32Array> {
  return (await embed([text], kind))[0];
}

export function cosine(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
