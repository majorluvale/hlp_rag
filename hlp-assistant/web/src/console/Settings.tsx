import { useEffect, useState } from 'react';
import { api } from '../api';

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
  { id: 'custom', label: 'Autre (compatible OpenAI)', baseURL: '', model: '', fastModel: '' },
];

export function Settings() {
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

  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm({ ...form, [k]: e.target.type === 'checkbox' ? (e.target as HTMLInputElement).checked : e.target.value });

  const choosePreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setForm({ ...form, provider: id, baseURL: p.baseURL || form.baseURL, model: p.model, fastModel: p.fastModel, apiKey: '' });
    setModels([]);
    setMsg(null);
  };

  const test = async () => {
    setBusy(true);
    setMsg({ kind: 'info', text: 'Test en cours…' });
    try {
      const r = await api<{ ok: boolean; latencyMs?: number; reply?: string; error?: string; models: string[] }>('/admin/settings/llm/test', {
        body: { baseURL: form.baseURL, apiKey: form.apiKey, model: form.model },
      });
      setModels(r.models ?? []);
      setMsg(
        r.ok
          ? { kind: 'ok', text: `Connexion réussie en ${((r.latencyMs ?? 0) / 1000).toFixed(1)} s — réponse : « ${r.reply} »${r.models?.length ? ` · ${r.models.length} modèles disponibles` : ''}` }
          : { kind: 'err', text: `Échec : ${r.error}${r.models?.length ? ` (la clé fonctionne : ${r.models.length} modèles listés, vérifiez le nom du modèle)` : ''}` },
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
      setMsg({ kind: 'ok', text: 'Paramètres enregistrés : ils s’appliquent immédiatement à toutes les nouvelles questions.' });
    } catch (e) {
      setMsg({ kind: 'err', text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    if (!confirm('Revenir à la configuration des variables d’environnement (.env / Vercel) ?')) return;
    const r = await api<{ llm: LlmSettings }>('/admin/settings/llm', { method: 'PUT', body: { reset: true } });
    fill(r.llm);
    setMsg({ kind: 'info', text: 'Configuration de la console supprimée : retour aux variables d’environnement.' });
  };

  if (!saved) return <div className="card pad">Chargement…</div>;
  const preset = PRESETS.find((p) => p.id === form.provider);

  return (
    <div className="settings">
      <section className="card pad">
        <div className="row between">
          <h3>Modèle de langage (API LLM)</h3>
          <span className={`badge ${saved.source === 'console' ? 'green' : 'grey'}`}>
            {saved.source === 'console' ? 'configuré ici' : 'depuis les variables d’environnement'}
          </span>
        </div>
        <p className="small muted">
          Toute API compatible OpenAI fonctionne. La clé est chiffrée dans la base de données et n’est jamais renvoyée au navigateur.
          Les changements s’appliquent immédiatement, sans redéploiement.
        </p>

        <form className="form grid2 flush" onSubmit={(e) => { e.preventDefault(); save(); }}>
          <label>
            Fournisseur
            <select value={form.provider} onChange={(e) => choosePreset(e.target.value)}>
              {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
          <label>
            URL de base de l’API
            <input value={form.baseURL} onChange={set('baseURL')} placeholder="https://…/v1" required />
          </label>
          <label className="span2">
            Clé API {preset?.keyUrl && <a className="xsmall" href={preset.keyUrl} target="_blank" rel="noreferrer">obtenir une clé</a>}
            <input
              type="password"
              autoComplete="off"
              value={form.apiKey}
              onChange={set('apiKey')}
              placeholder={saved.hasApiKey ? `Clé enregistrée ${saved.apiKeyHint} — laisser vide pour la conserver` : 'Collez la clé API'}
            />
          </label>
          {saved.hasApiKey && (
            <label className="span2 check">
              <input type="checkbox" checked={form.clearApiKey} onChange={set('clearApiKey')} /> Supprimer la clé enregistrée
            </label>
          )}
          <label>
            Modèle principal
            <input value={form.model} onChange={set('model')} list="llm-models" placeholder="ex. openai/gpt-oss-120b" required />
          </label>
          <label>
            Modèle rapide (évaluateur automatique)
            <input value={form.fastModel} onChange={set('fastModel')} list="llm-models" placeholder="vide = modèle principal" />
          </label>
          <datalist id="llm-models">{models.map((m) => <option key={m} value={m} />)}</datalist>
          <label>
            Température (0 = factuel, 1 = créatif)
            <input type="number" min={0} max={2} step={0.1} value={form.temperature} onChange={set('temperature')} />
          </label>
          <label>
            Étapes max. de l’agent (recherches successives)
            <input type="number" min={1} max={10} value={form.maxSteps} onChange={set('maxSteps')} />
          </label>

          {msg && <div className={`alert span2 ${msg.kind === 'ok' ? 'success' : msg.kind === 'info' ? 'info' : ''}`}>{msg.text}</div>}

          <div className="row between span2">
            <button type="button" className="btn ghost" onClick={reset} disabled={busy || saved.source !== 'console'}>Revenir au .env</button>
            <span className="row gap8">
              <button type="button" className="btn outline" onClick={test} disabled={busy || !form.baseURL}>Tester la connexion</button>
              <button className="btn primary" disabled={busy}>Enregistrer</button>
            </span>
          </div>
        </form>
      </section>

      {system && (
        <section className="card pad">
          <h3>Environnement</h3>
          <table className="table">
            <tbody>
              <tr><td>Base de données</td><td>{system.database === 'turso' ? 'Turso (hébergée)' : 'SQLite local (fichier data/hlp.db)'}</td></tr>
              <tr><td>Embeddings</td><td>{system.embedding.provider} — {system.embedding.model}</td></tr>
              <tr><td>Hébergement</td><td>{system.vercel ? 'Vercel (serverless)' : 'Serveur Node.js'}</td></tr>
              <tr><td>Miroir Langfuse</td><td>{system.langfuse ? 'activé' : 'désactivé'}</td></tr>
            </tbody>
          </table>
          <p className="xsmall muted">Ces éléments se règlent par variables d’environnement (voir README).</p>
        </section>
      )}
    </div>
  );
}
