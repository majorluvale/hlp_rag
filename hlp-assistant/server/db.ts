import { DatabaseSync, type StatementSync } from 'node:sqlite';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { config } from './config';

fs.mkdirSync(config.dataDir, { recursive: true });

export const db = new DatabaseSync(path.join(config.dataDir, 'hlp.db'));

db.exec(`
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE,
  display_name TEXT,
  role TEXT NOT NULL DEFAULT 'guest',        -- guest | user | reviewer | admin
  is_guest INTEGER NOT NULL DEFAULT 1,
  profile_json TEXT NOT NULL DEFAULT '{}',
  created_at INTEGER NOT NULL,
  last_seen_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
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
  user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at);

CREATE TABLE IF NOT EXISTS messages (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
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
  source_type TEXT NOT NULL,                 -- doc | web
  hash TEXT,
  pages INTEGER,
  chunks INTEGER NOT NULL DEFAULT 0,
  ingested_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
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
  status TEXT NOT NULL DEFAULT 'running',    -- running | ok | error
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
  trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
  type TEXT NOT NULL,                        -- retrieval | llm | tool | eval
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
  trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
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
  trace_id TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
  user_id TEXT,
  kind TEXT NOT NULL,                        -- user | reviewer | auto
  rating REAL NOT NULL,                      -- user: -1/1 ; reviewer & auto: 1..5
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
`);

type Param = string | number | bigint | null | undefined | boolean | Uint8Array;

const stmts = new Map<string, StatementSync>();
function stmt(sql: string): StatementSync {
  let s = stmts.get(sql);
  if (!s) {
    s = db.prepare(sql);
    stmts.set(sql, s);
  }
  return s;
}
function norm(params: Param[]) {
  return params.map((p) => (p === undefined ? null : typeof p === 'boolean' ? (p ? 1 : 0) : p)) as (
    | string
    | number
    | bigint
    | null
    | Uint8Array
  )[];
}

export const all = <T = any>(sql: string, ...params: Param[]): T[] => stmt(sql).all(...norm(params)) as T[];
export const get = <T = any>(sql: string, ...params: Param[]): T | undefined =>
  stmt(sql).get(...norm(params)) as T | undefined;
export const run = (sql: string, ...params: Param[]) => stmt(sql).run(...norm(params));

export function transaction<T>(fn: () => T): T {
  db.exec('BEGIN');
  try {
    const r = fn();
    db.exec('COMMIT');
    return r;
  } catch (e) {
    db.exec('ROLLBACK');
    throw e;
  }
}

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
export function fromBlob(b: Uint8Array): Float32Array {
  const copy = b.slice();
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
