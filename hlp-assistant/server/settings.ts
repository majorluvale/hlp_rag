import crypto from 'node:crypto';
import { config } from './config';
import { all, batch, now, q } from './db';

/**
 * Runtime settings editable by admins from the console (stored in the database, so they
 * work on serverless hosts where .env cannot be changed). Secrets are encrypted with
 * AES-256-GCM using a key derived from SESSION_SECRET and are never sent back to the browser.
 */

const SECRET_KEYS = new Set(['llm.api_key']);
const KEY = crypto.createHash('sha256').update(`${config.sessionSecret}:settings:v1`).digest();
const CACHE_MS = 15_000;

function encrypt(plain: string): string {
  const iv = crypto.randomBytes(12);
  const c = crypto.createCipheriv('aes-256-gcm', KEY, iv);
  const enc = Buffer.concat([c.update(plain, 'utf8'), c.final()]);
  return 'enc:v1:' + Buffer.concat([iv, c.getAuthTag(), enc]).toString('base64');
}

function decrypt(stored: string): string {
  if (!stored.startsWith('enc:v1:')) return stored;
  try {
    const raw = Buffer.from(stored.slice(7), 'base64');
    const d = crypto.createDecipheriv('aes-256-gcm', KEY, raw.subarray(0, 12));
    d.setAuthTag(raw.subarray(12, 28));
    return Buffer.concat([d.update(raw.subarray(28)), d.final()]).toString('utf8');
  } catch {
    console.warn('[settings] secret illisible (SESSION_SECRET modifié ?) — ressaisissez la clé dans la console.');
    return '';
  }
}

let cache: { at: number; values: Map<string, string> } | null = null;

export async function getSettings(): Promise<Map<string, string>> {
  if (cache && Date.now() - cache.at < CACHE_MS) return cache.values;
  const rows = await all<{ key: string; value: string }>('SELECT key, value FROM settings');
  const values = new Map(rows.map((r) => [r.key, SECRET_KEYS.has(r.key) ? decrypt(r.value) : r.value]));
  cache = { at: Date.now(), values };
  return values;
}

/** null or '' deletes the setting (back to the environment default). */
export async function setSettings(entries: Record<string, string | null | undefined>, by: string) {
  const stmts = Object.entries(entries)
    .filter(([, v]) => v !== undefined)
    .map(([k, v]) =>
      v === null || v === ''
        ? q('DELETE FROM settings WHERE key = ?', k)
        : q(
            'INSERT INTO settings (key, value, updated_at, updated_by) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at, updated_by = excluded.updated_by',
            k, SECRET_KEYS.has(k) ? encrypt(v!) : v, now(), by,
          ),
    );
  await batch(stmts);
  cache = null;
}

export interface LlmConfig {
  provider: string;
  baseURL: string;
  apiKey: string;
  model: string;
  fastModel: string;
  temperature: number;
  maxSteps: number;
  reasoningEffort: string;
  source: 'console' | 'env';
}

export async function getLlmConfig(): Promise<LlmConfig> {
  const s = await getSettings();
  const fromConsole = s.has('llm.base_url');
  return {
    provider: s.get('llm.provider') ?? (fromConsole ? 'custom' : 'env'),
    baseURL: s.get('llm.base_url') ?? config.llm.baseURL,
    apiKey: s.get('llm.api_key') ?? config.llm.apiKey,
    model: s.get('llm.model') ?? config.llm.model,
    fastModel: s.get('llm.fast_model') ?? s.get('llm.model') ?? config.llm.fastModel,
    temperature: Number(s.get('llm.temperature') ?? config.llm.temperature),
    maxSteps: Number(s.get('llm.max_steps') ?? config.llm.maxSteps),
    reasoningEffort: s.get('llm.reasoning_effort') ?? config.llm.reasoningEffort,
    source: fromConsole ? 'console' : 'env',
  };
}

export const maskSecret = (s: string) => (s ? `••••${s.slice(-4)}` : '');
