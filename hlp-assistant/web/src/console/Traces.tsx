import { useCallback, useEffect, useState } from 'react';
import { api, timeAgo } from '../api';
import { Markdown } from '../components/Markdown';

const FILTERS = [
  { id: 'all', label: 'Toutes' },
  { id: 'negative', label: 'À risque' },
  { id: 'unreviewed', label: 'Non revues' },
  { id: 'reviewed', label: 'Revues' },
  { id: 'web', label: 'Avec recherche web' },
  { id: 'error', label: 'Erreurs' },
  { id: 'playground', label: 'Playground' },
];

const LABELS = ['Exacte', 'Inexacte', 'Incomplète', 'Hallucination', 'Hors sujet', 'Mauvaise langue', 'Source manquante', 'Source non pertinente', 'Ton inadapté', 'Risque Do No Harm'];

export function Traces() {
  const [filter, setFilter] = useState('all');
  const [q, setQ] = useState('');
  const [list, setList] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(() => {
    api<{ traces: any[] }>(`/admin/traces?filter=${filter}&q=${encodeURIComponent(q)}`).then((r) => setList(r.traces));
  }, [filter, q]);
  useEffect(() => {
    const t = setTimeout(load, 250);
    return () => clearTimeout(t);
  }, [load]);

  return (
    <div className="traces">
      <section className="card trace-list">
        <div className="pad-sm">
          <input className="search" placeholder="Rechercher une question ou une réponse…" value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="chips small-chips">
            {FILTERS.map((f) => (
              <button key={f.id} className={`chip ${filter === f.id ? 'on' : ''}`} onClick={() => setFilter(f.id)}>{f.label}</button>
            ))}
          </div>
        </div>
        <ul>
          {list.map((t) => (
            <li key={t.id} className={selected === t.id ? 'active' : ''} onClick={() => setSelected(t.id)}>
              <div className="row between">
                <span className="xsmall muted">{t.userName || t.userEmail || (t.isGuest ? 'Invité' : '—')} · {timeAgo(t.createdAt)}</span>
                <span className="row gap4">
                  {t.status === 'error' && <span className="badge red">erreur</span>}
                  {t.usedWeb ? <span className="badge blue">web</span> : null}
                  {t.usedCorrections ? <span className="badge green">correction</span> : null}
                  {t.userVote != null && <span className={`badge ${t.userVote > 0 ? 'green' : 'red'}`}>{t.userVote > 0 ? '👍' : '👎'}</span>}
                  {t.autoScore != null && <span className={`badge ${t.autoScore < 3.5 ? 'orange' : 'grey'}`}>auto {Number(t.autoScore).toFixed(1)}</span>}
                  {t.reviewerScore != null && <span className="badge purple">★ {t.reviewerScore}</span>}
                </span>
              </div>
              <div className="trace-q">{t.question}</div>
              <div className="xsmall muted clamp2">{t.answer}</div>
            </li>
          ))}
          {list.length === 0 && <li className="muted small">Aucune trace.</li>}
        </ul>
      </section>
      <section className="card trace-detail">
        {selected ? <TraceDetail id={selected} onReviewed={load} /> : <p className="muted pad">Sélectionnez une trace pour l’inspecter et l’évaluer.</p>}
      </section>
    </div>
  );
}

function TraceDetail({ id, onReviewed }: { id: string; onReviewed: () => void }) {
  const [d, setD] = useState<any>(null);
  const [rating, setRating] = useState(0);
  const [labels, setLabels] = useState<string[]>([]);
  const [comment, setComment] = useState('');
  const [corrected, setCorrected] = useState('');
  const [badDocs, setBadDocs] = useState<string[]>([]);
  const [msg, setMsg] = useState('');

  const load = useCallback(() => api(`/admin/traces/${id}`).then(setD), [id]);
  useEffect(() => {
    setD(null);
    setRating(0);
    setLabels([]);
    setComment('');
    setCorrected('');
    setBadDocs([]);
    setMsg('');
    load();
  }, [id, load]);

  if (!d) return <p className="muted pad">Chargement…</p>;
  const t = d.trace;

  const submit = async () => {
    try {
      const r = await api<{ correctionId: string | null }>(`/admin/traces/${id}/review`, {
        body: { rating, labels, comment, correctedAnswer: corrected, badDocIds: badDocs },
      });
      setMsg(r.correctionId ? 'Revue enregistrée. La correction sera utilisée pour les questions similaires.' : 'Revue enregistrée. Les sources ont été renforcées/pénalisées.');
      load();
      onReviewed();
    } catch (e) {
      setMsg((e as Error).message);
    }
  };

  return (
    <div className="detail">
      <div className="row between">
        <h3>Trace {t.id}</h3>
        <span className="xsmall muted">
          {new Date(t.created_at).toLocaleString('fr-FR')} · {t.latency_ms ? `${(t.latency_ms / 1000).toFixed(1)} s` : ''} · {t.input_tokens}+{t.output_tokens} tokens · {t.model} · prompt {t.promptVersion ? `v${t.promptVersion}` : 'test'}
        </span>
      </div>

      <div className="block">
        <div className="muted xsmall upper">Question</div>
        <div className="question">{t.question}</div>
      </div>
      <details className="block">
        <summary className="muted xsmall upper">Contexte de l’utilisateur</summary>
        <pre>{t.user_context}</pre>
      </details>
      <div className="block">
        <div className="muted xsmall upper">Réponse {t.status === 'error' && <span className="badge red">{t.error}</span>}</div>
        <div className="answer-box"><Markdown text={t.answer ?? ''} anchor={`tr-${t.id}`} /></div>
      </div>

      <div className="block">
        <div className="muted xsmall upper">Sources proposées au modèle ({d.sources.length})</div>
        {d.sources.map((s: any) => (
          <div key={s.n} className={`src-row ${s.cited ? 'cited' : ''}`}>
            <span className="source-n">{s.n}</span>
            <div className="grow">
              <div className="small">
                <strong>{s.title}</strong> {s.page ? `p. ${s.page}` : ''} <span className="badge grey">{s.kind}</span> {s.cited ? <span className="badge green">citée</span> : null}
                {s.score != null && <span className="xsmall muted"> score {Number(s.score).toFixed(4)}</span>}
              </div>
              {s.url && <a className="xsmall" href={s.url} target="_blank" rel="noreferrer">{s.url}</a>}
              <details><summary className="xsmall muted">extrait</summary><p className="xsmall">{s.snippet}</p></details>
            </div>
            {s.doc_id && (
              <label className="xsmall nowrap" title="Pénalise ce document lors des prochaines recherches">
                <input type="checkbox" checked={badDocs.includes(s.doc_id)} onChange={(e) => setBadDocs(e.target.checked ? [...badDocs, s.doc_id] : badDocs.filter((x) => x !== s.doc_id))} /> non pertinente
              </label>
            )}
          </div>
        ))}
      </div>

      <div className="block">
        <div className="muted xsmall upper">Étapes ({d.spans.length})</div>
        <div className="timeline">
          {d.spans.map((s: any) => (
            <details key={s.id} className={`span ${s.type} ${s.error ? 'err' : ''}`}>
              <summary>
                <span className={`badge ${s.type === 'llm' ? 'purple' : s.type === 'tool' ? 'blue' : 'grey'}`}>{s.type}</span> {s.name}
                <span className="xsmall muted"> · {s.duration_ms} ms</span>
                {s.error && <span className="badge red">{s.error}</span>}
              </summary>
              <div className="span-body">
                <div className="xsmall muted">Entrée</div>
                <pre>{JSON.stringify(s.input, null, 2)}</pre>
                <div className="xsmall muted">Sortie</div>
                <pre>{JSON.stringify(s.output, null, 2)}</pre>
                {s.meta && (<><div className="xsmall muted">Méta</div><pre>{JSON.stringify(s.meta, null, 2)}</pre></>)}
              </div>
            </details>
          ))}
        </div>
      </div>

      {d.feedback.length > 0 && (
        <div className="block">
          <div className="muted xsmall upper">Évaluations</div>
          {d.feedback.map((f: any) => (
            <div key={f.id} className="fb">
              <span className={`badge ${f.kind === 'reviewer' ? 'purple' : f.kind === 'auto' ? 'grey' : f.rating > 0 ? 'green' : 'red'}`}>
                {f.kind === 'user' ? (f.rating > 0 ? '👍 utilisateur' : '👎 utilisateur') : f.kind === 'auto' ? `auto ${f.rating}/5` : `équipe ★ ${f.rating}/5`}
              </span>
              <span className="xsmall muted"> {f.userName || f.userEmail || ''} · {timeAgo(f.created_at)}</span>
              {f.labels && <div className="xsmall">{Array.isArray(f.labels) ? f.labels.join(', ') : Object.entries(f.labels).map(([k, v]) => `${k}: ${v}`).join(' · ')}</div>}
              {f.comment && <div className="small">« {f.comment} »</div>}
              {f.corrected_answer && <details><summary className="xsmall">Réponse corrigée</summary><Markdown text={f.corrected_answer} /></details>}
            </div>
          ))}
        </div>
      )}

      <div className="block review card-inset">
        <h4>Évaluer et corriger</h4>
        <div className="stars">
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} className={`star ${n <= rating ? 'on' : ''}`} onClick={() => setRating(n)} aria-label={`${n} étoiles`}>★</button>
          ))}
          <span className="xsmall muted">{rating ? `${rating}/5` : 'Note requise'}</span>
        </div>
        <div className="chips small-chips">
          {LABELS.map((l) => (
            <button key={l} className={`chip ${labels.includes(l) ? 'on' : ''}`} onClick={() => setLabels(labels.includes(l) ? labels.filter((x) => x !== l) : [...labels, l])}>{l}</button>
          ))}
        </div>
        <textarea placeholder="Commentaire pour l’équipe" value={comment} onChange={(e) => setComment(e.target.value)} />
        <div className="row between">
          <span className="xsmall muted">Réponse corrigée (sera réutilisée par l’assistant pour les questions similaires)</span>
          <button className="btn ghost sm" onClick={() => setCorrected(t.answer ?? '')}>Partir de la réponse actuelle</button>
        </div>
        <textarea className="tall" placeholder="Rédigez la réponse attendue…" value={corrected} onChange={(e) => setCorrected(e.target.value)} />
        <div className="row end">
          {msg && <span className="small muted grow">{msg}</span>}
          <button className="btn primary" disabled={!rating} onClick={submit}>Enregistrer la revue</button>
        </div>
      </div>
    </div>
  );
}
