import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { createClient as createWebClient } from '@libsql/client/web';
import { config } from './config';

/**
 * Storage layer with two interchangeable drivers:
 *  - local file SQLite (node:sqlite) — default for development / a single server
 *  - libSQL / Turso (hosted, SQLite-compatible) — required on serverless hosts such as Vercel
 * Both speak the same SQL, so the rest of the app does not care which one runs.
 */

const SCHEMA = `
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE,
  display_name TEXT,
  role TEXT NOT NULL DEFAULT 'guest',
  is_guest INTEGER NOT NULL DEFAULT 1,
  profile_json TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE TABLE IF NOT EXISTS otp_codes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL,
  code_hash TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0,
  consumed INTEGER NOT NULL DEFAULT 0,
  expires_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_otp_email ON otp_codes(email, created_at);
CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  title TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  trace_id TEXT,
  sources_json TEXT,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at);
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,
  path TEXT,
  url TEXT,
  title TEXT NOT NULL,
  source_type TEXT NOT NULL,
  hash TEXT,
  pages INTEGER,
  chunks INTEGER NOT NULL DEFAULT 0,
  ingested_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL,
  page INTEGER,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS source_rewards (
  doc_id TEXT PRIMARY KEY,
  alpha REAL NOT NULL DEFAULT 1,
  beta REAL NOT NULL DEFAULT 1,
  updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS prompts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version INTEGER NOT NULL,
  content TEXT NOT NULL,
  note TEXT,
  author TEXT,
  is_active INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS traces (
  id TEXT PRIMARY KEY,
  user_id TEXT,
  conversation_id TEXT,
  question TEXT NOT NULL,
  answer TEXT,
  user_context TEXT,
  prompt_version_id INTEGER,
  model TEXT,
  status TEXT NOT NULL DEFAULT 'running',
  error TEXT,
  used_web INTEGER NOT NULL DEFAULT 0,
  used_corrections INTEGER NOT NULL DEFAULT 0,
  latency_ms INTEGER,
  input_tokens INTEGER DEFAULT 0,
  output_tokens INTEGER DEFAULT 0,
  is_playground INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);
CREATE TABLE IF NOT EXISTS spans (
  id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  type TEXT NOT NULL,
  name TEXT NOT NULL,
  input_json TEXT,
  output_json TEXT,
  meta_json TEXT,
  error TEXT,
  start_at INTEGER NOT NULL,
  duration_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id, start_at);
CREATE TABLE IF NOT EXISTS trace_sources (
  trace_id TEXT NOT NULL,
  n INTEGER NOT NULL,
  kind TEXT NOT NULL,
  doc_id TEXT,
  chunk_id INTEGER,
  title TEXT,
  url TEXT,
  page INTEGER,
  score REAL,
  cited INTEGER NOT NULL DEFAULT 0,
  snippet TEXT,
  PRIMARY KEY (trace_id, n)
);
CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY,
  trace_id TEXT NOT NULL,
  user_id TEXT,
  kind TEXT NOT NULL,
  rating REAL NOT NULL,
  labels_json TEXT,
  comment TEXT,
  corrected_answer TEXT,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_trace ON feedback(trace_id);
CREATE TABLE IF NOT EXISTS corrections (
  id TEXT PRIMARY KEY,
  trace_id TEXT,
  question TEXT NOT NULL,
  corrected_answer TEXT NOT NULL,
  embedding BLOB NOT NULL,
  author_id TEXT,
  active INTEGER NOT NULL DEFAULT 1,
  hits INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS web_cache (
  url TEXT PRIMARY KEY,
  title TEXT,
  text TEXT NOT NULL,
  fetched_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  updated_by TEXT
);
`;

type Arg = string | number | bigint | null | Uint8Array;
export type Param = Arg | undefined | boolean;
export interface Stmt {
  sql: string;
  args: Param[];
}

interface Driver {
  kind: 'sqlite' | 'turso';
  all(sql: string, args: Arg[]): Promise<Record<string, unknown>[]>;
  run(sql: string, args: Arg[]): Promise<{ lastInsertRowid: number; changes: number }>;
  /** Runs several write statements atomically. */
  batch(stmts: { sql: string; args: Arg[] }[]): Promise<void>;
  exec(sql: string): Promise<void>;
}

async function sqliteDriver(file: string): Promise<Driver> {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const { DatabaseSync } = await import('node:sqlite');
  const db = new DatabaseSync(file);
  db.exec('PRAGMA journal_mode = WAL; PRAGMA busy_timeout = 5000;');
  const cache = new Map<string, ReturnType<typeof db.prepare>>();
  const st = (sql: string) => {
    let s = cache.get(sql);
    if (!s) cache.set(sql, (s = db.prepare(sql)));
    return s;
  };
  return {
    kind: 'sqlite',
    async all(sql, args) {
      return st(sql).all(...args) as Record<string, unknown>[];
    },
    async run(sql, args) {
      const r = st(sql).run(...args);
      return { lastInsertRowid: Number(r.lastInsertRowid), changes: Number(r.changes) };
    },
    async batch(stmts) {
      db.exec('BEGIN');
      try {
        for (const s of stmts) st(s.sql).run(...s.args);
        db.exec('COMMIT');
      } catch (e) {
        db.exec('ROLLBACK');
        throw e;
      }
    },
    async exec(sql) {
      db.exec(sql);
    },
  };
}

async function tursoDriver(url: string, authToken: string): Promise<Driver> {
  type Client = ReturnType<typeof createWebClient>;
  let client: Client;
  if (url.startsWith('file:')) {
    // local libSQL file (tests): native client, never bundled for the serverless build
    const spec = ['@libsql', 'client'].join('/');
    const mod = (await import(spec)) as { createClient: typeof createWebClient };
    client = mod.createClient({ url });
  } else {
    client = createWebClient({ url, authToken: authToken || undefined });
  }
  const conv = (v: unknown) => (v instanceof ArrayBuffer ? new Uint8Array(v) : v);
  return {
    kind: 'turso',
    async all(sql, args) {
      const r = await client.execute({ sql, args });
      return r.rows.map((row) => Object.fromEntries(r.columns.map((c, i) => [c, conv(row[i])])));
    },
    async run(sql, args) {
      const r = await client.execute({ sql, args });
      return { lastInsertRowid: Number(r.lastInsertRowid ?? 0), changes: r.rowsAffected };
    },
    async batch(stmts) {
      if (stmts.length) await client.batch(stmts, 'write');
    },
    async exec(sql) {
      await client.executeMultiple(sql);
    },
  };
}

let driver: Driver | null = null;
let initPromise: Promise<void> | null = null;

export function initDb(): Promise<void> {
  initPromise ??= (async () => {
    if (config.db.url) {
      driver = await tursoDriver(config.db.url, config.db.authToken);
    } else {
      if (config.isVercel) throw new Error('Sur Vercel, configurez DATABASE_URL et DATABASE_AUTH_TOKEN (Turso).');
      driver = await sqliteDriver(path.join(config.dataDir, 'hlp.db'));
    }
    await driver.exec(SCHEMA);
  })();
  return initPromise;
}

function d(): Driver {
  if (!driver) throw new Error('Base de données non initialisée');
  return driver;
}

function norm(params: Param[]): Arg[] {
  return params.map((p) => (p === undefined ? null : typeof p === 'boolean' ? (p ? 1 : 0) : p));
}

export const dbKind = () => d().kind;
export const all = async <T = any>(sql: string, ...params: Param[]): Promise<T[]> => (await d().all(sql, norm(params))) as T[];
export const get = async <T = any>(sql: string, ...params: Param[]): Promise<T | undefined> => (await all<T>(sql, ...params))[0];
export const run = (sql: string, ...params: Param[]) => d().run(sql, norm(params));
export const q = (sql: string, ...args: Param[]): Stmt => ({ sql, args });
export const batch = (stmts: Stmt[]) => d().batch(stmts.map((s) => ({ sql: s.sql, args: norm(s.args) })));

export const now = () => Date.now();
export const uid = (prefix = '') => prefix + crypto.randomUUID().replace(/-/g, '').slice(0, 20);

export function parseJson<T>(s: unknown, fallback: T): T {
  if (typeof s !== 'string' || !s) return fallback;
  try {
    return JSON.parse(s) as T;
  } catch {
    return fallback;
  }
}

export function toBlob(v: Float32Array): Uint8Array {
  return new Uint8Array(v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength));
}
export function fromBlob(b: Uint8Array | ArrayBuffer): Float32Array {
  const copy = b instanceof ArrayBuffer ? new Uint8Array(b.slice(0)) : b.slice();
  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

export interface UserRow {
  id: string;
  email: string | null;
  display_name: string | null;
  role: 'guest' | 'user' | 'reviewer' | 'admin';
  is_guest: number;
  profile_json: string;
  created_at: number;
  last_seen_at: number;
}

export interface UserProfile {
  jobTitle?: string;
  organisation?: string;
  sector?: string;
  country?: string;
  expertise?: string;
  language?: string;
  interests?: string;
}
