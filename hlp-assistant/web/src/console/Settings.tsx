import { useEffect, useState, type ChangeEvent } from 'react';
import { api } from '../api';
import { useI18n } from '../i18n';

interface LlmSettings {
  provider: string;
  baseURL: string;
  model: string;
  fastModel: string;
  temperature: number;
  maxSteps: number;
  hasApiKey: boolean;
  apiKeyHint: string;
  source: 'console' | 'env';
}

/** OpenAI-compatible providers. Models are suggestions; "Load models" lists what the key really gives access to. */
const PRESETS: { id: string; label: string; baseURL: string; model: string; fastModel: string; keyUrl?: string }[] = [
  { id: 'groq', label: 'Groq', baseURL: 'https://api.groq.com/openai/v1', model: 'openai/gpt-oss-120b', fastModel: 'openai/gpt-oss-20b', keyUrl: 'https://console.groq.com/keys' },
  { id: 'openai', label: 'OpenAI', baseURL: 'https://api.openai.com/v1', model: '', fastModel: '', keyUrl: 'https://platform.openai.com/api-keys' },
  { id: 'anthropic', label: 'Anthropic (Claude)', baseURL: 'https://api.anthropic.com/v1', model: '', fastModel: '', keyUrl: 'https://console.anthropic.com/settings/keys' },
  { id: 'mistral', label: 'Mistral AI', baseURL: 'https://api.mistral.ai/v1', model: 'mistral-large-latest', fastModel: 'mistral-small-latest', keyUrl: 'https://console.mistral.ai/api-keys' },
  { id: 'openrouter', label: 'OpenRouter', baseURL: 'https://openrouter.ai/api/v1', model: '', fastModel: '', keyUrl: 'https://openrouter.ai/keys' },
  { id: 'azure', label: 'Azure OpenAI', baseURL: 'https://VOTRE-RESSOURCE.openai.azure.com/openai/v1', model: '', fastModel: '' },
  { id: 'ollama', label: 'Ollama (local)', baseURL: 'http://localhost:11434/v1', model: 'llama3.1', fastModel: 'llama3.1' },
  { id: 'custom', label: '', baseURL: '', model: '', fastModel: '' },
];

export function Settings() {
  const { t } = useI18n();
  const [saved, setSaved] = useState<LlmSettings | null>(null);
  const [system, setSystem] = useState<any>(null);
  const [form, setForm] = useState({ provider: 'groq', baseURL: '', apiKey: '', clearApiKey: false, model: '', fastModel: '', temperature: '0.3', maxSteps: '6' });
  const [models, setModels] = useState<string[]>([]);
  const [msg, setMsg] = useState<{ kind: 'ok' | 'err' | 'info'; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const fill = (s: LlmSettings) => {
    setSaved(s);
    const known = PRESETS.find((p) => p.id === s.provider) ?? PRESETS.find((p) => p.baseURL && s.baseURL.startsWith(p.baseURL));
    setForm({
      provider: known?.id ?? 'custom',
      baseURL: s.baseURL,
      apiKey: '',
      clearApiKey: false,
      model: s.model,
      fastModel: s.fastModel,
      temperature: String(s.temperature),
      maxSteps: String(s.maxSteps),
    });
  };

  useEffect(() => {
    api<{ llm: LlmSettings; system: any }>('/admin/settings').then((r) => {
      fill(r.llm);
      setSystem(r.system);
    });
  }, []);

  const set = (k: keyof typeof form) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.type === 'checkbox' ? (e.target as HTMLInputElement).checked : e.target.value });

  const choosePreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setForm({ ...form, provider: id, baseURL: p.baseURL || form.baseURL, model: p.model, fastModel: p.fastModel, apiKey: '' });
    setModels([]);
    setMsg(null);
  };

  const test = async () => {
    setBusy(true);
    setMsg({ kind: 'info', text: t('set.testing') });
    try {
      const r = await api<{ ok: boolean; latencyMs?: number; reply?: string; error?: string; models: string[] }>('/admin/settings/llm/test', {
        body: { baseURL: form.baseURL, apiKey: form.apiKey, model: form.model },
      });
      setModels(r.models ?? []);
      setMsg(
        r.ok
          ? { kind: 'ok', text: t('set.ok', { s: ((r.latencyMs ?? 0) / 1000).toFixed(1), reply: r.reply ?? '', models: r.models?.length ? t('set.models', { n: r.models.length }) : '' }) }
          : { kind: 'err', text: t('set.fail', { error: r.error ?? '', models: r.models?.length ? t('set.failModels', { n: r.models.length }) : '' }) },
      );
    } catch (e) {
      setMsg({ kind: 'err', text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    setBusy(true);
    try {
      const r = await api<{ llm: LlmSettings }>('/admin/settings/llm', {
        method: 'PUT',
        body: {
          provider: form.provider,
          baseURL: form.baseURL,
          apiKey: form.apiKey,
          clearApiKey: form.clearApiKey,
          model: form.model,
          fastModel: form.fastModel,
          temperature: Number(form.temperature),
          maxSteps: Number(form.maxSteps),
        },
      });
      fill(r.llm);
      setMsg({ kind: 'ok', text: t('set.saved') });
    } catch (e) {
      setMsg({ kind: 'err', text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    if (!confirm(t('set.confirmReset'))) return;
    const r = await api<{ llm: LlmSettings }>('/admin/settings/llm', { method: 'PUT', body: { reset: true } });
    fill(r.llm);
    setMsg({ kind: 'info', text: t('set.resetDone') });
  };

  if (!saved) return <div className="card pad">{t('common.loading')}</div>;
  const preset = PRESETS.find((p) => p.id === form.provider);

  return (
    <div className="settings">
      <section className="card pad">
        <div className="row between">
          <h3>{t('set.llm')}</h3>
          <span className={`badge ${saved.source === 'console' ? 'green' : 'grey'}`}>
            {saved.source === 'console' ? t('set.fromConsole') : t('set.fromEnv')}
          </span>
        </div>
        <p className="small muted">{t('set.intro')}</p>

        <form className="form grid2 flush" onSubmit={(e) => { e.preventDefault(); save(); }}>
          <label>
            {t('set.provider')}
            <select value={form.provider} onChange={(e) => choosePreset(e.target.value)}>
              {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label || t('set.custom')}</option>)}
            </select>
          </label>
          <label>
            {t('set.baseUrl')}
            <input value={form.baseURL} onChange={set('baseURL')} placeholder="https://…/v1" required />
          </label>
          <label className="span2">
            {t('set.apiKey')} {preset?.keyUrl && <a className="xsmall" href={preset.keyUrl} target="_blank" rel="noreferrer">{t('set.getKey')}</a>}
            <input
              type="password"
              autoComplete="off"
              value={form.apiKey}
              onChange={set('apiKey')}
              placeholder={saved.hasApiKey ? t('set.keySaved', { hint: saved.apiKeyHint }) : t('set.keyPaste')}
            />
          </label>
          {saved.hasApiKey && (
            <label className="span2 check">
              <input type="checkbox" checked={form.clearApiKey} onChange={set('clearApiKey')} /> {t('set.clearKey')}
            </label>
          )}
          <label>
            {t('set.model')}
            <input value={form.model} onChange={set('model')} list="llm-models" placeholder="ex. openai/gpt-oss-120b" required />
          </label>
          <label>
            {t('set.fastModel')}
            <input value={form.fastModel} onChange={set('fastModel')} list="llm-models" placeholder={t('set.fastModelPh')} />
          </label>
          <datalist id="llm-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
          <label>
            {t('set.temperature')}
            <input type="number" min={0} max={2} step={0.1} value={form.temperature} onChange={set('temperature')} />
          </label>
          <label>
            {t('set.maxSteps')}
            <input type="number" min={1} max={10} value={form.maxSteps} onChange={set('maxSteps')} />
          </label>

          {msg && <div className={`alert span2 ${msg.kind === 'ok' ? 'success' : msg.kind === 'info' ? 'info' : ''}`}>{msg.text}</div>}

          <div className="row between span2">
            <button type="button" className="btn ghost" onClick={reset} disabled={busy || saved.source !== 'console'}>{t('set.backToEnv')}</button>
            <span className="row gap8">
              <button type="button" className="btn outline" onClick={test} disabled={busy || !form.baseURL}>{t('set.test')}</button>
              <button className="btn primary" disabled={busy}>{t('common.save')}</button>
            </span>
          </div>
        </form>
      </section>

      {system && (
        <section className="card pad">
          <h3>{t('set.env')}</h3>
          <table className="table">
            <tbody>
              <tr><td>{t('set.database')}</td><td>{system.database === 'turso' ? t('set.dbTurso') : t('set.dbLocal')}</td></tr>
              <tr><td>{t('set.embeddings')}</td><td>{system.embedding.provider} — {system.embedding.model}</td></tr>
              <tr><td>{t('set.hosting')}</td><td>{system.vercel ? t('set.vercel') : t('set.node')}</td></tr>
              <tr><td>{t('set.langfuse')}</td><td>{system.langfuse ? t('set.on') : t('set.off')}</td></tr>
            </tbody>
          </table>
          <p className="xsmall muted">{t('set.envHint')}</p>
        </section>
      )}
    </div>
  );
}
