import { useCallback, useEffect, useState } from 'react';
import { api } from '../api';
import { timeAgoT, useI18n } from '../i18n';
import { Markdown } from '../components/Markdown';

const FILTERS = ['all', 'negative', 'unreviewed', 'reviewed', 'web', 'error', 'playground'] as const;

export function Traces() {
  const { t, lang } = useI18n();
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>('all');
  const [q, setQ] = useState('');
  const [list, setList] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(() => {
    api<{ traces: any[] }>(`/admin/traces?filter=${filter}&q=${encodeURIComponent(q)}`).then((r) => setList(r.traces));
  }, [filter, q]);
  useEffect(() => {
    const timer = setTimeout(load, 250);
    return () => clearTimeout(timer);
  }, [load]);

  return (
    <div className="traces">
      <section className="card trace-list">
        <div className="pad-sm">
          <input className="search" placeholder={t('traces.search')} value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="chips small-chips">
            {FILTERS.map((f) => (
              <button key={f} className={`chip ${filter === f ? 'on' : ''}`} onClick={() => setFilter(f)}>{t(`traces.f.${f}`)}</button>
            ))}
          </div>
        </div>
        <ul>
          {list.map((tr) => (
            <li key={tr.id} className={selected === tr.id ? 'active' : ''} onClick={() => setSelected(tr.id)}>
              <div className="row between">
                <span className="xsmall muted">{tr.userName || tr.userEmail || (tr.isGuest ? t('common.guest') : '—')} · {timeAgoT(t, tr.createdAt, lang)}</span>
                <span className="row gap4">
                  {tr.status === 'error' && <span className="badge red">{t('traces.error')}</span>}
                  {tr.usedWeb ? <span className="badge blue">web</span> : null}
                  {tr.usedCorrections ? <span className="badge green">{t('traces.correction')}</span> : null}
                  {tr.userVote != null && <span className={`badge ${tr.userVote > 0 ? 'green' : 'red'}`}>{tr.userVote > 0 ? '👍' : '👎'}</span>}
                  {tr.autoScore != null && <span className={`badge ${tr.autoScore < 3.5 ? 'orange' : 'grey'}`}>auto {Number(tr.autoScore).toFixed(1)}</span>}
                  {tr.reviewerScore != null && <span className="badge purple">★ {tr.reviewerScore}</span>}
                </span>
              </div>
              <div className="trace-q">{tr.question}</div>
              <div className="xsmall muted clamp2">{tr.answer}</div>
            </li>
          ))}
          {list.length === 0 && <li className="muted small">{t('traces.none')}</li>}
        </ul>
      </section>
      <section className="card trace-detail">
        {selected ? <TraceDetail id={selected} onReviewed={load} /> : <p className="muted pad">{t('traces.select')}</p>}
      </section>
    </div>
  );
}

function TraceDetail({ id, onReviewed }: { id: string; onReviewed: () => void }) {
  const { t, lang } = useI18n();
  const labelsList = t('traces.labels').split('|');
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

  if (!d) return <p className="muted pad">{t('common.loading')}</p>;
  const tr = d.trace;

  const submit = async () => {
    try {
      const r = await api<{ correctionId: string | null }>(`/admin/traces/${id}/review`, {
        body: { rating, labels, comment, correctedAnswer: corrected, badDocIds: badDocs },
      });
      setMsg(r.correctionId ? t('traces.savedWithCorrection') : t('traces.saved'));
      load();
      onReviewed();
    } catch (e) {
      setMsg((e as Error).message);
    }
  };

  return (
    <div className="detail">
      <div className="row between">
        <h3>Trace {tr.id}</h3>
        <span className="xsmall muted">
          {new Date(tr.created_at).toLocaleString(lang === 'fr' ? 'fr-FR' : 'en-GB')} · {tr.latency_ms ? `${(tr.latency_ms / 1000).toFixed(1)} s` : ''} · {tr.input_tokens}+{tr.output_tokens} tokens · {tr.model} ·{' '}
          {t('traces.prompt')} {tr.promptVersion ? `v${tr.promptVersion}` : t('traces.test')}
        </span>
      </div>

      <div className="block">
        <div className="muted xsmall upper">{t('traces.question')}</div>
        <div className="question">{tr.question}</div>
      </div>
      <details className="block">
        <summary className="muted xsmall upper">{t('traces.userContext')}</summary>
        <pre>{tr.user_context}</pre>
      </details>
      <div className="block">
        <div className="muted xsmall upper">{t('traces.answer')} {tr.status === 'error' && <span className="badge red">{tr.error}</span>}</div>
        <div className="answer-box"><Markdown text={tr.answer ?? ''} anchor={`tr-${tr.id}`} /></div>
      </div>

      <div className="block">
        <div className="muted xsmall upper">{t('traces.sources', { n: d.sources.length })}</div>
        {d.sources.map((s: any) => (
          <div key={s.n} className={`src-row ${s.cited ? 'cited' : ''}`}>
            <span className="source-n">{s.n}</span>
            <div className="grow">
              <div className="small">
                <strong>{s.title}</strong> {s.page ? `p. ${s.page}` : ''} <span className="badge grey">{s.kind}</span> {s.cited ? <span className="badge green">{t('traces.cited')}</span> : null}
                {s.score != null && <span className="xsmall muted"> score {Number(s.score).toFixed(4)}</span>}
              </div>
              {s.url && <a className="xsmall" href={s.url} target="_blank" rel="noreferrer">{s.url}</a>}
              <details><summary className="xsmall muted">{t('traces.excerpt')}</summary><p className="xsmall">{s.snippet}</p></details>
            </div>
            {s.doc_id && (
              <label className="xsmall nowrap" title={t('traces.irrelevantHint')}>
                <input type="checkbox" checked={badDocs.includes(s.doc_id)} onChange={(e) => setBadDocs(e.target.checked ? [...badDocs, s.doc_id] : badDocs.filter((x) => x !== s.doc_id))} /> {t('traces.irrelevant')}
              </label>
            )}
          </div>
        ))}
      </div>

      <div className="block">
        <div className="muted xsmall upper">{t('traces.steps', { n: d.spans.length })}</div>
        <div className="timeline">
          {d.spans.map((s: any) => (
            <details key={s.id} className={`span ${s.type} ${s.error ? 'err' : ''}`}>
              <summary>
                <span className={`badge ${s.type === 'llm' ? 'purple' : s.type === 'tool' ? 'blue' : 'grey'}`}>{s.type}</span> {s.name}
                <span className="xsmall muted"> · {s.duration_ms} ms</span>
                {s.error && <span className="badge red">{s.error}</span>}
              </summary>
              <div className="span-body">
                <div className="xsmall muted">{t('traces.input')}</div>
                <pre>{JSON.stringify(s.input, null, 2)}</pre>
                <div className="xsmall muted">{t('traces.output')}</div>
                <pre>{JSON.stringify(s.output, null, 2)}</pre>
                {s.meta && (<><div className="xsmall muted">{t('traces.meta')}</div><pre>{JSON.stringify(s.meta, null, 2)}</pre></>)}
              </div>
            </details>
          ))}
        </div>
      </div>

      {d.feedback.length > 0 && (
        <div className="block">
          <div className="muted xsmall upper">{t('traces.evaluations')}</div>
          {d.feedback.map((f: any) => (
            <div key={f.id} className="fb">
              <span className={`badge ${f.kind === 'reviewer' ? 'purple' : f.kind === 'auto' ? 'grey' : f.rating > 0 ? 'green' : 'red'}`}>
                {f.kind === 'user' ? (f.rating > 0 ? t('traces.userUp') : t('traces.userDown')) : f.kind === 'auto' ? t('traces.auto', { n: f.rating }) : t('traces.team', { n: f.rating })}
              </span>
              <span className="xsmall muted"> {f.userName || f.userEmail || ''} · {timeAgoT(t, f.created_at, lang)}</span>
              {f.labels && <div className="xsmall">{Array.isArray(f.labels) ? f.labels.join(', ') : Object.entries(f.labels).map(([k, v]) => `${k}: ${v}`).join(' · ')}</div>}
              {f.comment && <div className="small">“{f.comment}”</div>}
              {f.corrected_answer && <details><summary className="xsmall">{t('traces.correctedAnswer')}</summary><Markdown text={f.corrected_answer} /></details>}
            </div>
          ))}
        </div>
      )}

      <div className="block review card-inset">
        <h4>{t('traces.review')}</h4>
        <div className="stars">
          {[1, 2, 3, 4, 5].map((n) => (
            <button key={n} className={`star ${n <= rating ? 'on' : ''}`} onClick={() => setRating(n)} aria-label={`${n}/5`}>★</button>
          ))}
          <span className="xsmall muted">{rating ? `${rating}/5` : t('traces.ratingRequired')}</span>
        </div>
        <div className="chips small-chips">
          {labelsList.map((l) => (
            <button key={l} className={`chip ${labels.includes(l) ? 'on' : ''}`} onClick={() => setLabels(labels.includes(l) ? labels.filter((x) => x !== l) : [...labels, l])}>{l}</button>
          ))}
        </div>
        <textarea placeholder={t('traces.comment')} value={comment} onChange={(e) => setComment(e.target.value)} />
        <div className="row between">
          <span className="xsmall muted">{t('traces.correctedHint')}</span>
          <button className="btn ghost sm" onClick={() => setCorrected(tr.answer ?? '')}>{t('traces.fromCurrent')}</button>
        </div>
        <textarea className="tall" placeholder={t('traces.writeExpected')} value={corrected} onChange={(e) => setCorrected(e.target.value)} />
        <div className="row end">
          {msg && <span className="small muted grow">{msg}</span>}
          <button className="btn primary" disabled={!rating} onClick={submit}>{t('traces.saveReview')}</button>
        </div>
      </div>
    </div>
  );
}
