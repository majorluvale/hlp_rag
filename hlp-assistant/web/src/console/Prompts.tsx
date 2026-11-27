import { useEffect, useMemo, useState } from 'react';
import { api, timeAgo, type Source } from '../api';
import { Markdown } from '../components/Markdown';

interface PromptRow {
  id: number;
  name: string;
  version: number;
  content: string;
  note: string | null;
  author: string | null;
  is_active: number;
  created_at: number;
}

const VARS: Record<string, string[]> = {
  system: ['{{knowledge}}', '{{corrections}}', '{{user_context}}', '{{date}}', '{{web_domains}}'],
  judge: ['{{question}}', '{{answer}}', '{{sources}}'],
};

export function Prompts() {
  const [rows, setRows] = useState<PromptRow[]>([]);
  const [name, setName] = useState<'system' | 'judge'>('system');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [draft, setDraft] = useState('');
  const [note, setNote] = useState('');
  const [msg, setMsg] = useState('');
  const [testQ, setTestQ] = useState('');
  const [testing, setTesting] = useState(false);
  const [test, setTest] = useState<{ answer: string; sources: Source[]; traceId: string } | null>(null);

  const load = () => api<{ prompts: PromptRow[] }>('/admin/prompts').then((r) => setRows(r.prompts));
  useEffect(() => { load(); }, []);

  const versions = useMemo(() => rows.filter((r) => r.name === name), [rows, name]);
  const selected = versions.find((v) => v.id === selectedId) ?? versions.find((v) => v.is_active) ?? versions[0];

  useEffect(() => {
    if (selected) {
      setSelectedId(selected.id);
      setDraft(selected.content);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name, rows.length]);

  const save = async (activate: boolean) => {
    try {
      const r = await api<{ prompt: PromptRow }>('/admin/prompts', { body: { name, content: draft, note, activate } });
      setMsg(`Version v${r.prompt.version} enregistrée${activate ? ' et activée' : ''}.`);
      setNote('');
      await load();
      setSelectedId(r.prompt.id);
    } catch (e) {
      setMsg((e as Error).message);
    }
  };

  const activate = async (id: number) => {
    await api(`/admin/prompts/${id}/activate`, { body: {} });
    setMsg('Version activée : elle est utilisée immédiatement pour les nouvelles questions.');
    load();
  };

  const runTest = async () => {
    setTesting(true);
    setTest(null);
    try {
      setTest(await api('/admin/playground', { body: { question: testQ, content: name === 'system' ? draft : undefined } }));
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setTesting(false);
    }
  };

  const dirty = selected && draft !== selected.content;

  return (
    <div className="prompts">
      <section className="card versions">
        <div className="pad-sm">
          <div className="seg">
            <button className={name === 'system' ? 'on' : ''} onClick={() => { setName('system'); setSelectedId(null); }}>Prompt système</button>
            <button className={name === 'judge' ? 'on' : ''} onClick={() => { setName('judge'); setSelectedId(null); }}>Évaluateur (juge)</button>
          </div>
        </div>
        <ul>
          {versions.map((v) => (
            <li key={v.id} className={v.id === selected?.id ? 'active' : ''} onClick={() => { setSelectedId(v.id); setDraft(v.content); }}>
              <div className="row between">
                <strong>v{v.version}</strong>
                {v.is_active ? <span className="badge green">active</span> : <button className="btn ghost sm" onClick={(e) => { e.stopPropagation(); activate(v.id); }}>Activer</button>}
              </div>
              <div className="xsmall muted">{v.note || 'Sans note'} · {v.author} · {timeAgo(v.created_at)}</div>
            </li>
          ))}
        </ul>
      </section>

      <section className="card editor">
        <div className="row between">
          <h3>{name === 'system' ? 'Prompt système de l’assistant' : 'Prompt de l’évaluateur automatique'} {selected && <span className="muted small">— basé sur v{selected.version}</span>}</h3>
          {dirty && <span className="badge orange">modifié</span>}
        </div>
        <div className="xsmall muted">Variables disponibles : {VARS[name].map((v) => <code key={v}>{v}</code>)}</div>
        <textarea className="code" value={draft} onChange={(e) => setDraft(e.target.value)} spellCheck={false} />
        <div className="row gap8">
          <input className="grow" placeholder="Note de version (ex. « réponses plus courtes pour le terrain »)" value={note} onChange={(e) => setNote(e.target.value)} />
          <button className="btn outline" disabled={!dirty} onClick={() => save(false)}>Enregistrer en brouillon</button>
          <button className="btn primary" disabled={!dirty} onClick={() => save(true)}>Enregistrer et activer</button>
        </div>
        {msg && <div className="alert info">{msg}</div>}

        {name === 'system' && (
          <div className="playground card-inset">
            <h4>Tester ce prompt (sans l’activer)</h4>
            <div className="row gap8">
              <input className="grow" placeholder="Question de test…" value={testQ} onChange={(e) => setTestQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && testQ && runTest()} />
              <button className="btn primary" disabled={!testQ || testing} onClick={runTest}>{testing ? 'Exécution…' : 'Tester'}</button>
            </div>
            {test && (
              <div className="answer-box">
                <Markdown text={test.answer} sources={test.sources} anchor="pg" />
                <div className="xsmall muted">Trace : {test.traceId} (filtre « Playground » dans Traces)</div>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
