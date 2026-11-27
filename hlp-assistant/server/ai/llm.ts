import OpenAI from 'openai';
import { getLlmConfig, type LlmConfig } from '../settings';

/**
 * Any OpenAI-compatible endpoint: Groq (default), OpenAI, Mistral, OpenRouter, Anthropic,
 * Azure OpenAI, Ollama, vLLM… Configured in the console (Settings) or in .env.
 */
let cached: { key: string; client: OpenAI } | null = null;

export function makeClient(baseURL: string, apiKey: string) {
  return new OpenAI({ baseURL, apiKey: apiKey || 'missing-key', timeout: 120_000, maxRetries: 1 });
}

export async function getLlm(): Promise<{ client: OpenAI; cfg: LlmConfig }> {
  const cfg = await getLlmConfig();
  const key = `${cfg.baseURL}|${cfg.apiKey}`;
  if (!cached || cached.key !== key) cached = { key, client: makeClient(cfg.baseURL, cfg.apiKey) };
  return { client: cached.client, cfg };
}
