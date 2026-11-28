import { getLang } from './i18n';

export interface Profile {
  jobTitle?: string;
  organisation?: string;
  sector?: string;
  country?: string;
  expertise?: string;
  language?: string;
  interests?: string;
}

export interface User {
  id: string;
  email: string | null;
  displayName: string | null;
  role: 'guest' | 'user' | 'reviewer' | 'admin';
  isGuest: boolean;
  profile: Profile;
}

export interface Source {
  n: number;
  kind: 'doc' | 'web' | 'correction';
  title: string;
  url?: string | null;
  page?: number | null;
  snippet: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  traceId?: string | null;
  sources?: Source[];
  vote?: number;
  createdAt?: number;
  pending?: boolean;
  status?: string;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  updatedAt: number;
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const res = await fetch('/api' + path, {
    method: opts.method ?? (opts.body ? 'POST' : 'GET'),
    headers: { 'X-Lang': getLang(), ...(opts.body ? { 'Content-Type': 'application/json' } : {}) },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    credentials: 'same-origin',
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error ?? `Erreur ${res.status}`);
  return data as T;
}

export type StreamEvent =
  | { type: 'conversation'; conversationId: string }
  | { type: 'status'; text: string }
  | { type: 'token'; text: string }
  | { type: 'reset' }
  | { type: 'sources'; sources: Source[] }
  | { type: 'done'; traceId: string; messageId: string }
  | { type: 'error'; error: string; traceId?: string };

export async function streamChat(
  body: { message: string; conversationId: string | null },
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Lang': getLang() },
    body: JSON.stringify({ ...body, lang: getLang() }),
    credentials: 'same-origin',
    signal,
  });
  if (!res.ok || !res.body) {
    const d = await res.json().catch(() => ({}));
    throw new Error(d.error ?? `Erreur ${res.status}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i: number;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      for (const line of block.split('\n')) {
        if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      }
    }
  }
}

export const initials = (u: Pick<User, 'displayName' | 'email'> | null, fallback = '?') => {
  const s = u?.displayName || u?.email || fallback;
  return s
    .split(/[\s@._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join('');
};
