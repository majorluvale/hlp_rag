import { useEffect, useState } from 'react';
import { api, timeAgo } from '../api';
import { Markdown } from '../components/Markdown';

export function Corrections() {
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
      <h3>Mémoire des corrections</h3>
      <p className="small muted">
        Chaque réponse corrigée par l’équipe est indexée. Quand une question similaire arrive, la correction est injectée en priorité dans le contexte de l’assistant.
      </p>
      {rows.length === 0 && <p className="muted">Aucune correction pour l’instant. Corrigez une réponse depuis « Traces & évaluation ».</p>}
      {rows.map((c) => (
        <div key={c.id} className={`correction ${c.active ? '' : 'off'}`}>
          <div className="row between">
            <strong>{c.question}</strong>
            <span className="row gap8">
              <span className="xsmall muted">{c.hits} réutilisations · {c.author || c.authorEmail} · {timeAgo(c.createdAt)}</span>
              <button className="btn ghost sm" onClick={() => { setEditing(c.id); setText(c.correctedAnswer); }}>Modifier</button>
              <button className={`btn sm ${c.active ? 'outline' : 'primary'}`} onClick={() => patch(c.id, { active: !c.active })}>{c.active ? 'Désactiver' : 'Réactiver'}</button>
            </span>
          </div>
          {editing === c.id ? (
            <>
              <textarea className="tall" value={text} onChange={(e) => setText(e.target.value)} />
              <div className="row end"><button className="btn ghost" onClick={() => setEditing(null)}>Annuler</button><button className="btn primary" onClick={() => patch(c.id, { correctedAnswer: text })}>Enregistrer</button></div>
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
  const [d, setD] = useState<any>(null);
  const load = () => api('/admin/knowledge').then(setD);
  useEffect(() => {
    load();
    const t = setInterval(() => d?.status?.running && load(), 3000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [d?.status?.running]);

  if (!d) return <div className="card pad">Chargement…</div>;
  const st = d.status;
  return (
    <section className="card pad">
      <div className="row between">
        <h3>Base de connaissances</h3>
        <button className="btn primary" disabled={st.running || !d.canIngest} title={d.canIngest ? '' : 'Sur Vercel : lancez « npm run ingest » depuis votre PC'} onClick={async () => { await api('/admin/knowledge/reindex', { body: {} }); load(); }}>
          {st.running ? `Indexation ${st.done}/${st.total}…` : 'Indexer les nouveaux documents'}
        </button>
      </div>
      <p className="small muted">
        Déposez des PDF dans les dossiers configurés (DOCS_DIRS) puis lancez l’indexation. Les pages web officielles lues par l’assistant sont apprises automatiquement.
        Le score de fiabilité de chaque source évolue avec les retours (renforcement).
      </p>
      {!d.canIngest && <div className="alert info">Version en ligne (Vercel) : ajoutez des documents en lançant <code>npm run ingest</code> sur votre PC, avec <code>DATABASE_URL</code> pointant vers la base Turso.</div>}
      {d.indexEmbedding && !d.indexEmbedding.startsWith(d.embedding.family + ':') && (
        <div className="alert">L’index a été construit avec « {d.indexEmbedding} » mais le modèle d’embeddings actuel est « {d.embedding.model} ». Ré-indexez ou utilisez la même famille de modèle.</div>
      )}
      {st.running && <div className="alert info">En cours : {st.current}</div>}
      {st.errors?.length > 0 && <div className="alert">{st.errors.join(' · ')}</div>}
      <table className="table">
        <thead><tr><th>Document</th><th>Type</th><th>Extraits</th><th>Citations</th><th>Fiabilité apprise</th>{isAdmin && <th />}</tr></thead>
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
                  <div className="meter" title={`α=${doc.alpha.toFixed(1)} β=${doc.beta.toFixed(1)}`}>
                    <div style={{ width: `${r * 100}%`, background: r >= 0.55 ? 'var(--green)' : r <= 0.45 ? 'var(--red)' : 'var(--muted)' }} />
                  </div>
                </td>
                {isAdmin && <td><button className="btn ghost sm" onClick={async () => { if (confirm('Retirer ce document de la base ?')) { await api(`/admin/knowledge/${doc.id}`, { method: 'DELETE' }); load(); } }}>Retirer</button></td>}
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

export function Users({ isAdmin }: { isAdmin: boolean }) {
  const [rows, setRows] = useState<any[]>([]);
  const load = () => api<{ users: any[] }>('/admin/users').then((r) => setRows(r.users));
  useEffect(() => { load(); }, []);
  return (
    <section className="card pad">
      <h3>Utilisateurs</h3>
      <p className="small muted">Les relecteurs (reviewer) accèdent à l’observabilité. Les administrateurs gèrent les rôles et la base documentaire.</p>
      <table className="table">
        <thead><tr><th>Nom</th><th>Email</th><th>Contexte</th><th>Questions</th><th>Vu</th><th>Rôle</th></tr></thead>
        <tbody>
          {rows.map((u) => (
            <tr key={u.id}>
              <td>{u.displayName || '—'}</td>
              <td>{u.email || <span className="muted">invité</span>}</td>
              <td className="xsmall">{[u.profile.jobTitle, u.profile.organisation, u.profile.country].filter(Boolean).join(' · ')}</td>
              <td>{u.questions}</td>
              <td className="xsmall">{timeAgo(u.lastSeenAt)}</td>
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
