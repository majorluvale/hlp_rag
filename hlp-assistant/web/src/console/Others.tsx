import { useEffect, useState } from 'react';
import { api } from '../api';
import { timeAgoT, useI18n } from '../i18n';
import { Markdown } from '../components/Markdown';

export function Corrections() {
  const { t, lang } = useI18n();
  const [rows, setRows] = useState<any[]>([]);
  const [editing, setEditing] = useState<string | null>(null);
  const [text, setText] = useState('');
  const load = () => api<{ corrections: any[] }>('/admin/corrections').then((r) => setRows(r.corrections));
  useEffect(() => { load(); }, []);

  const patch = async (id: string, body: object) => {
    await api(`/admin/corrections/${id}`, { method: 'PATCH', body });
    setEditing(null);
    load();
  };

  return (
    <section className="card pad">
      <h3>{t('corr.title')}</h3>
      <p className="small muted">{t('corr.intro')}</p>
      {rows.length === 0 && <p className="muted">{t('corr.empty')}</p>}
      {rows.map((c) => (
        <div key={c.id} className={`correction ${c.active ? '' : 'off'}`}>
          <div className="row between">
            <strong>{c.question}</strong>
            <span className="row gap8">
              <span className="xsmall muted">{t('corr.reuses', { n: c.hits })} · {c.author || c.authorEmail} · {timeAgoT(t, c.createdAt, lang)}</span>
              <button className="btn ghost sm" onClick={() => { setEditing(c.id); setText(c.correctedAnswer); }}>{t('common.edit')}</button>
              <button className={`btn sm ${c.active ? 'outline' : 'primary'}`} onClick={() => patch(c.id, { active: !c.active })}>{c.active ? t('corr.disable') : t('corr.enable')}</button>
            </span>
          </div>
          {editing === c.id ? (
            <>
              <textarea className="tall" value={text} onChange={(e) => setText(e.target.value)} />
              <div className="row end"><button className="btn ghost" onClick={() => setEditing(null)}>{t('common.cancel')}</button><button className="btn primary" onClick={() => patch(c.id, { correctedAnswer: text })}>{t('common.save')}</button></div>
            </>
          ) : (
            <Markdown text={c.correctedAnswer} />
          )}
        </div>
      ))}
    </section>
  );
}

export function Knowledge({ isAdmin }: { isAdmin: boolean }) {
  const { t } = useI18n();
  const [d, setD] = useState<any>(null);
  const load = () => api('/admin/knowledge').then(setD);
  useEffect(() => {
    load();
    const timer = setInterval(() => d?.status?.running && load(), 3000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d?.status?.running]);

  if (!d) return <div className="card pad">{t('common.loading')}</div>;
  const st = d.status;
  return (
    <section className="card pad">
      <div className="row between">
        <h3>{t('kb.title')}</h3>
        <button className="btn primary" disabled={st.running || !d.canIngest} title={d.canIngest ? '' : t('kb.vercelHint')} onClick={async () => { await api('/admin/knowledge/reindex', { body: {} }); load(); }}>
          {st.running ? t('kb.indexing', { done: st.done, total: st.total }) : t('kb.index')}
        </button>
      </div>
      <p className="small muted">{t('kb.intro')}</p>
      {!d.canIngest && <div className="alert info">{t('kb.vercel')}</div>}
      {d.indexEmbedding && !d.indexEmbedding.startsWith(d.embedding.family + ':') && (
        <div className="alert">{t('kb.mismatch', { idx: d.indexEmbedding, model: d.embedding.model })}</div>
      )}
      {st.running && <div className="alert info">{t('kb.running', { f: st.current ?? '' })}</div>}
      {st.errors?.length > 0 && <div className="alert">{st.errors.join(' · ')}</div>}
      <table className="table">
        <thead><tr><th>{t('kb.doc')}</th><th>{t('kb.type')}</th><th>{t('kb.extracts')}</th><th>{t('kb.citations')}</th><th>{t('kb.reliability')}</th>{isAdmin && <th />}</tr></thead>
        <tbody>
          {d.documents.map((doc: any) => {
            const r = doc.alpha / (doc.alpha + doc.beta);
            return (
              <tr key={doc.id}>
                <td>{doc.url ? <a href={doc.url} target="_blank" rel="noreferrer">{doc.title}</a> : doc.title}</td>
                <td><span className={`badge ${doc.sourceType === 'web' ? 'blue' : 'grey'}`}>{doc.sourceType}</span></td>
                <td>{doc.chunks}</td>
                <td>{doc.citations}</td>
                <td>
                  <div className="meter" title={`α=${Number(doc.alpha).toFixed(1)} β=${Number(doc.beta).toFixed(1)}`}>
                    <div style={{ width: `${r * 100}%`, background: r >= 0.55 ? 'var(--green)' : r <= 0.45 ? 'var(--red)' : 'var(--muted)' }} />
                  </div>
                </td>
                {isAdmin && <td><button className="btn ghost sm" onClick={async () => { if (confirm(t('kb.confirmRemove'))) { await api(`/admin/knowledge/${doc.id}`, { method: 'DELETE' }); load(); } }}>{t('kb.remove')}</button></td>}
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

export function Users({ isAdmin }: { isAdmin: boolean }) {
  const { t, lang } = useI18n();
  const [rows, setRows] = useState<any[]>([]);
  const load = () => api<{ users: any[] }>('/admin/users').then((r) => setRows(r.users));
  useEffect(() => { load(); }, []);
  return (
    <section className="card pad">
      <h3>{t('users.title')}</h3>
      <p className="small muted">{t('users.intro')}</p>
      <table className="table">
        <thead><tr><th>{t('users.name')}</th><th>{t('users.email')}</th><th>{t('users.context')}</th><th>{t('users.questions')}</th><th>{t('users.seen')}</th><th>{t('users.role')}</th></tr></thead>
        <tbody>
          {rows.map((u) => (
            <tr key={u.id}>
              <td>{u.displayName || '—'}</td>
              <td>{u.email || <span className="muted">{t('common.guest')}</span>}</td>
              <td className="xsmall">{[u.profile.jobTitle, u.profile.organisation, u.profile.country].filter(Boolean).join(' · ')}</td>
              <td>{u.questions}</td>
              <td className="xsmall">{timeAgoT(t, u.lastSeenAt, lang)}</td>
              <td>
                {isAdmin && !u.isGuest ? (
                  <select value={u.role} onChange={async (e) => { await api(`/admin/users/${u.id}`, { method: 'PATCH', body: { role: e.target.value } }); load(); }}>
                    <option value="user">user</option>
                    <option value="reviewer">reviewer</option>
                    <option value="admin">admin</option>
                  </select>
                ) : (
                  <span className="badge grey">{u.role}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
