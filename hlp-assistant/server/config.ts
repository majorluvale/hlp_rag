import 'dotenv/config';
import path from 'node:path';

const root = process.cwd();

function list(value: string | undefined, fallback: string[]): string[] {
  if (!value) return fallback;
  return value
    .split(/[;,]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export const config = {
  port: Number(process.env.PORT ?? 3000),
  dataDir: path.resolve(root, process.env.DATA_DIR ?? (process.env.VERCEL ? '/tmp/hlp' : 'data')),
  docsDirs: list(process.env.DOCS_DIRS, ['data/docs', '../../docs', '../../knowledge']).map((d) =>
    path.resolve(root, d),
  ),
  autoIngest: process.env.AUTO_INGEST !== 'false',

  llm: {
    baseURL: process.env.LLM_BASE_URL ?? 'https://api.groq.com/openai/v1',
    apiKey: process.env.LLM_API_KEY ?? process.env.GROQ_API_KEY ?? '',
    model: process.env.LLM_MODEL ?? 'openai/gpt-oss-120b',
    fastModel: process.env.LLM_FAST_MODEL ?? 'openai/gpt-oss-20b',
    temperature: Number(process.env.LLM_TEMPERATURE ?? 0.3),
    maxSteps: Number(process.env.AGENT_MAX_STEPS ?? 6),
    /** reasoning models (gpt-oss, o-series, gpt-5): low = much faster answers, plenty for answering from retrieved sources */
    reasoningEffort: process.env.LLM_REASONING_EFFORT ?? 'low',
  },

  isVercel: Boolean(process.env.VERCEL),

  /** Turso / libSQL. Empty = local SQLite file in DATA_DIR. */
  db: {
    url: process.env.DATABASE_URL ?? process.env.TURSO_DATABASE_URL ?? '',
    authToken: process.env.DATABASE_AUTH_TOKEN ?? process.env.TURSO_AUTH_TOKEN ?? '',
  },

  /** local = transformers.js on CPU ; huggingface = HF Inference API (same e5 model, for Vercel) ; openai = any OpenAI-compatible /embeddings */
  embedding: {
    provider: (process.env.EMBEDDING_PROVIDER ?? 'local') as 'local' | 'huggingface' | 'openai',
    model: process.env.EMBEDDING_MODEL ?? '',
    hfToken: process.env.HF_TOKEN ?? '',
    baseURL: process.env.EMBEDDING_BASE_URL ?? 'https://api.openai.com/v1',
    apiKey: process.env.EMBEDDING_API_KEY ?? '',
  },

  sessionSecret: process.env.SESSION_SECRET ?? 'dev-secret-change-me',
  adminEmails: list(process.env.ADMIN_EMAILS, []).map((e) => e.toLowerCase()),
  devShowOtp: process.env.DEV_SHOW_OTP === 'true',

  smtp: {
    host: process.env.SMTP_HOST ?? '',
    port: Number(process.env.SMTP_PORT ?? 587),
    user: process.env.SMTP_USER ?? '',
    pass: process.env.SMTP_PASS ?? '',
    from: process.env.SMTP_FROM ?? 'HLP Assistant <no-reply@example.org>',
  },

  web: {
    tavilyKey: process.env.TAVILY_API_KEY ?? '',
    domains: list(process.env.WEB_DOMAINS, [
      'globalprotectioncluster.org',
      'nrc.no',
      'unhabitat.org',
      'cccmcluster.org',
      'sheltercluster.org',
      'reliefweb.int',
    ]),
    learnFromWeb: process.env.LEARN_FROM_WEB !== 'false',
  },

  langfuse: {
    publicKey: process.env.LANGFUSE_PUBLIC_KEY ?? '',
    secretKey: process.env.LANGFUSE_SECRET_KEY ?? '',
    baseUrl: process.env.LANGFUSE_BASE_URL ?? 'https://cloud.langfuse.com',
  },

  autoEval: process.env.AUTO_EVAL !== 'false',
  guestHourlyLimit: Number(process.env.GUEST_HOURLY_LIMIT ?? 30),
};

export type AppConfig = typeof config;
