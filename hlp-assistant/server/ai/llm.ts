import OpenAI from 'openai';
import { config } from '../config';

/** Any OpenAI-compatible endpoint: Groq (default), OpenAI, Azure OpenAI, Mistral, Ollama, vLLM… */
export const llm = new OpenAI({
  baseURL: config.llm.baseURL,
  apiKey: config.llm.apiKey || 'missing-key',
});

export const llmConfigured = () => Boolean(config.llm.apiKey);
