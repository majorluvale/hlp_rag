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
    headers: opts.body ? { 'Content-Type': 'application/json' } : undefined,
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
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

export const initials = (u: Pick<User, 'displayName' | 'email'> | null) => {
  const s = u?.displayName || u?.email || 'Invité';
  return s
    .split(/[\s@._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]!.toUpperCase())
    .join('');
};

export const timeAgo = (ms: number) => {
  const d = (Date.now() - ms) / 1000;
  if (d < 60) return "à l'instant";
  if (d < 3600) return `${Math.floor(d / 60)} min`;
  if (d < 86400) return `${Math.floor(d / 3600)} h`;
  if (d < 86400 * 7) return `${Math.floor(d / 86400)} j`;
  return new Date(ms).toLocaleDateString('fr-FR');
};
