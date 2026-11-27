import { useEffect, useRef, useState } from 'react';
import { api, initials, streamChat, type ChatMessage, type Conversation, type Source, type User } from '../api';
import { ACRONYM, timeAgoT, useI18n } from '../i18n';
import { Markdown } from './Markdown';
import {
  IconBook, IconCheck, IconCopy, IconEdit, IconGlobe, IconPlus, IconSend, IconSpark, IconThumbDown, IconThumbUp, IconTrash,
} from './Icons';

const SUGGESTIONS = ['suggest.1', 'suggest.2', 'suggest.3', 'suggest.4', 'suggest.5'] as const;

interface Props {
  user: User | null;
  conversations: Conversation[];
  refreshConversations: () => Promise<void>;
  onLogin: () => void;
  onEditContext: () => void;
}

export function ChatPage({ user, conversations, refreshConversations, onLogin, onEditContext }: Props) {
  const { t, lang } = useI18n();
  const ago = (ms?: number) => (ms ? timeAgoT(t, ms, lang) : '');
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState<{ stats: { docs: number; web: number; corrections: number } } | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const guestLabel = t('common.guest');
  const acronym = ACRONYM[lang];

  useEffect(() => {
    api('/info').then(setInfo).catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages]);

  const openConversation = async (id: string) => {
    abortRef.current?.abort();
    const r = await api<{ messages: ChatMessage[] }>(`/conversations/${id}`);
    setConversationId(id);
    setMessages(r.messages);
  };

  const newConversation = () => {
    abortRef.current?.abort();
    setConversationId(null);
    setMessages([]);
  };

  const deleteConversation = async (id: string) => {
    if (!confirm(t('conv.confirmDelete'))) return;
    await api(`/conversations/${id}`, { method: 'DELETE' });
    if (id === conversationId) newConversation();
    await refreshConversations();
  };

  const patchLast = (fn: (m: ChatMessage) => ChatMessage) => setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));

  const send = async (text: string) => {
    const message = text.trim();
    if (!message || busy) return;
    setBusy(true);
    const tmpId = 'tmp' + Date.now();
    setMessages((ms) => [
      ...ms,
      { id: tmpId + 'u', role: 'user', content: message, createdAt: Date.now() },
      { id: tmpId, role: 'assistant', content: '', pending: true, status: t('status.thinking'), createdAt: Date.now() },
    ]);
    const ac = new AbortController();
    abortRef.current = ac;
    let convId = conversationId;
    try {
      await streamChat(
        { message, conversationId },
        (e) => {
          if (e.type === 'conversation') {
            convId = e.conversationId;
            setConversationId(e.conversationId);
          } else if (e.type === 'status') patchLast((m) => ({ ...m, status: e.text }));
          else if (e.type === 'token') patchLast((m) => ({ ...m, content: m.content + e.text, status: undefined }));
          else if (e.type === 'reset') patchLast((m) => ({ ...m, content: '' }));
          else if (e.type === 'sources') patchLast((m) => ({ ...m, sources: e.sources }));
          else if (e.type === 'done') patchLast((m) => ({ ...m, id: e.messageId, traceId: e.traceId, pending: false, status: undefined }));
          else if (e.type === 'error') patchLast((m) => ({ ...m, pending: false, status: undefined, error: e.error }));
        },
        ac.signal,
      );
    } catch (err) {
      if (!ac.signal.aborted) patchLast((m) => ({ ...m, pending: false, status: undefined, error: (err as Error).message }));
    } finally {
      setBusy(false);
      if (convId) refreshConversations();
    }
  };

  const vote = async (m: ChatMessage, rating: 1 | -1, comment?: string) => {
    if (!m.traceId) return;
    await api('/feedback', { body: { traceId: m.traceId, rating, comment } });
    setMessages((ms) => ms.map((x) => (x.id === m.id ? { ...x, vote: rating } : x)));
  };

  const p = user?.profile ?? {};
  const contextFilled = Boolean(p.jobTitle || p.organisation || p.country);

  return (
    <main className="layout">
      {/* ---------- Left rail: identity & conversations ---------- */}
      <aside className="rail left">
        <section className="card profile-card">
          <div className="cover" />
          <div className="avatar lg">{initials(user, guestLabel)}</div>
          <h2>{user?.displayName || (user?.isGuest ? guestLabel : user?.email)}</h2>
          <p className="muted">
            {[p.jobTitle, p.organisation].filter(Boolean).join(' · ') || (user?.isGuest ? t('profile.guestSubtitle') : t('profile.toComplete'))}
          </p>
          {p.country && <p className="muted small">{p.country}</p>}
          <div className="divider" />
          <button className="link-row" onClick={onEditContext}>
            <span>{contextFilled ? t('profile.myContext') : t('profile.personalise')}</span>
            <IconEdit width={16} height={16} />
          </button>
          {user?.isGuest && (
            <div className="profile-cta">
              <p className="small muted">{t('profile.signInCta')}</p>
              <button className="btn primary block" onClick={onLogin}>{t('menu.signIn')}</button>
            </div>
          )}
        </section>

        <section className="card conv-card">
          <div className="card-head">
            <h3>{t('conv.title')}</h3>
            <button className="icon-btn" title={t('conv.new')} onClick={newConversation}>
              <IconPlus />
            </button>
          </div>
          {conversations.length === 0 && <p className="muted small pad">{t('conv.empty')}</p>}
          <ul className="conv-list">
            {conversations.map((c) => (
              <li key={c.id} className={c.id === conversationId ? 'active' : ''}>
                <button onClick={() => openConversation(c.id)} title={c.title}>
                  <span className="conv-title">{c.title}</span>
                  <span className="muted xsmall">{ago(c.updatedAt)}</span>
                </button>
                <button className="icon-btn ghost" title={t('conv.delete')} onClick={() => deleteConversation(c.id)}>
                  <IconTrash width={15} height={15} />
                </button>
              </li>
            ))}
          </ul>
        </section>
      </aside>

      {/* ---------- Feed ---------- */}
      <section className="feed">
        {messages.length === 0 && (
          <article className="card welcome">
            <div className="post-head">
              <span className="avatar md brand-avatar">{acronym}</span>
              <div>
                <strong>{t('assistant.name')}</strong>
                <div className="muted small">{t('assistant.subtitle')}</div>
              </div>
            </div>
            <p className="welcome-hello">{t('welcome.hello', { name: user?.displayName ? ` ${user.displayName}` : '' })}</p>
            <Markdown text={t('welcome.body')} />
            <p className="muted small">{t('welcome.body2')}</p>
            <div className="muted xsmall upper try">{t('welcome.try')}</div>
            <div className="chips">
              {SUGGESTIONS.map((k) => (
                <button key={k} className="chip" onClick={() => send(t(k))}>
                  {t(k)}
                </button>
              ))}
            </div>
          </article>
        )}

        {messages.map((m) =>
          m.role === 'user' ? (
            <article key={m.id} className="card post user-post">
              <div className="post-head">
                <span className="avatar sm">{initials(user, guestLabel)}</span>
                <div>
                  <strong>{user?.displayName || t('common.you')}</strong>
                  <div className="muted xsmall">{ago(m.createdAt)}</div>
                </div>
              </div>
              <p className="question">{m.content}</p>
            </article>
          ) : (
            <AnswerCard key={m.id} m={m} onVote={vote} />
          ),
        )}
        <div ref={bottomRef} />

        <div className="composer card">
          <span className="avatar sm">{initials(user, guestLabel)}</span>
          <Composer busy={busy} onSend={send} onStop={() => abortRef.current?.abort()} />
        </div>
      </section>

      {/* ---------- Right rail ---------- */}
      <aside className="rail right">
        <section className="card">
          <div className="card-head">
            <h3>{t('rail.whatIsTitle')}</h3>
            <IconBook width={16} height={16} />
          </div>
          <p className="small">{t('rail.whatIsText')}</p>
        </section>
        <section className="card">
          <div className="card-head">
            <h3>{t('rail.learning')}</h3>
            <IconSpark width={16} height={16} />
          </div>
          <div className="stats">
            <div><strong>{info?.stats.docs ?? '–'}</strong><span className="muted xsmall">{t('rail.docs')}</span></div>
            <div><strong>{info?.stats.web ?? '–'}</strong><span className="muted xsmall">{t('rail.web')}</span></div>
            <div><strong>{info?.stats.corrections ?? '–'}</strong><span className="muted xsmall">{t('rail.corrections')}</span></div>
          </div>
          <p className="small muted">{t('rail.learningText')}</p>
        </section>
        <footer className="foot muted xsmall">{t('rail.footer')}</footer>
      </aside>
    </main>
  );
}

function Composer({ busy, onSend, onStop }: { busy: boolean; onSend: (t: string) => void; onStop: () => void }) {
  const { t } = useI18n();
  const [text, setText] = useState('');
  const ref = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [text]);
  const submit = () => {
    if (!text.trim()) return;
    onSend(text);
    setText('');
  };
  return (
    <div className="composer-body">
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder={t('composer.placeholder')}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submit();
          }
        }}
      />
      {busy ? (
        <button className="btn outline" onClick={onStop}>{t('composer.stop')}</button>
      ) : (
        <button className="btn primary round" onClick={submit} disabled={!text.trim()} title={t('common.send')}>
          <IconSend width={18} height={18} />
        </button>
      )}
    </div>
  );
}

function SourceIcon({ s }: { s: Source }) {
  if (s.kind === 'web') return <IconGlobe width={15} height={15} />;
  if (s.kind === 'correction') return <IconCheck width={15} height={15} />;
  return <IconBook width={15} height={15} />;
}

function AnswerCard({ m, onVote }: { m: ChatMessage; onVote: (m: ChatMessage, r: 1 | -1, c?: string) => Promise<void> }) {
  const { t, lang } = useI18n();
  const [copied, setCopied] = useState(false);
  const [askWhy, setAskWhy] = useState(false);
  const [why, setWhy] = useState('');
  const [thanks, setThanks] = useState(false);

  return (
    <article className="card post answer">
      <div className="post-head">
        <span className="avatar sm brand-avatar">{ACRONYM[lang]}</span>
        <div>
          <strong>{t('assistant.name')}</strong>
          <div className="muted xsmall">{t('assistant.short')} · {m.createdAt ? timeAgoT(t, m.createdAt, lang) : ''}</div>
        </div>
      </div>

      {m.status && (
        <div className="status">
          <span className="spinner" /> {m.status}
        </div>
      )}
      {m.content && <Markdown text={m.content} sources={m.sources} anchor={m.id} />}
      {m.pending && m.content && <span className="caret" />}
      {m.error && <div className="alert">{m.error}</div>}

      {!!m.sources?.length && (
        <div className="sources">
          <div className="muted xsmall upper">{t('answer.sources')}</div>
          {m.sources.map((s) => (
            <div key={s.n} id={`${m.id}-src-${s.n}`} className={`source ${s.kind}`}>
              <span className="source-n">{s.n}</span>
              <SourceIcon s={s} />
              {s.url ? (
                <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a>
              ) : (
                <span title={s.snippet.slice(0, 400)}>{s.kind === 'correction' ? t('answer.validated') : s.title}</span>
              )}
              {s.page ? <span className="muted xsmall">p. {s.page}</span> : null}
              {s.kind === 'correction' && <span className="badge green">{t('answer.validated')}</span>}
            </div>
          ))}
        </div>
      )}

      {!m.pending && m.traceId && (
        <>
          <div className="actions">
            <button className={`action ${m.vote === 1 ? 'on' : ''}`} onClick={() => { onVote(m, 1); setThanks(true); setAskWhy(false); }}>
              <IconThumbUp width={18} height={18} /> {t('answer.useful')}
            </button>
            <button className={`action ${m.vote === -1 ? 'on neg' : ''}`} onClick={() => setAskWhy(true)}>
              <IconThumbDown width={18} height={18} /> {t('answer.improve')}
            </button>
            <button
              className="action"
              onClick={() => {
                navigator.clipboard.writeText(m.content);
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }}
            >
              <IconCopy width={18} height={18} /> {copied ? t('answer.copied') : t('answer.copy')}
            </button>
          </div>
          {askWhy && (
            <div className="why">
              <textarea placeholder={t('answer.whyPlaceholder')} value={why} onChange={(e) => setWhy(e.target.value)} />
              <div className="row end">
                <button className="btn ghost" onClick={() => setAskWhy(false)}>{t('common.cancel')}</button>
                <button className="btn primary" onClick={async () => { await onVote(m, -1, why || undefined); setAskWhy(false); setThanks(true); }}>{t('common.send')}</button>
              </div>
            </div>
          )}
          {thanks && !askWhy && <div className="muted xsmall thanks">{t('answer.thanks')}</div>}
        </>
      )}
    </article>
  );
}
