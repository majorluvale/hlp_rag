import { useEffect, useState } from 'react';
import { api, type User } from '../api';
import { Traces } from './Traces';
import { Prompts } from './Prompts';
import { Corrections, Knowledge, Users } from './Others';

type Tab = 'dashboard' | 'traces' | 'corrections' | 'prompts' | 'knowledge' | 'users';
const TABS: { id: Tab; label: string }[] = [
  { id: 'dashboard', label: 'Tableau de bord' },
  { id: 'traces', label: 'Traces & évaluation' },
  { id: 'corrections', label: 'Corrections' },
  { id: 'prompts', label: 'Prompts' },
  { id: 'knowledge', label: 'Connaissances' },
  { id: 'users', label: 'Utilisateurs' },
];

const readTab = (): Tab => {
  const t = location.hash.split('/')[2] as Tab;
  return TABS.some((x) => x.id === t) ? t : 'dashboard';
};

export function Console({ user }: { user: User }) {
  const [tab, setTab] = useState<Tab>(readTab());
  useEffect(() => {
    const on = () => setTab(readTab());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);

  return (
    <main className="console">
      <div className="card console-head">
        <div>
          <h1>Observabilité & amélioration continue</h1>
          <p className="muted small">Inspectez chaque réponse, notez-la, corrigez-la : les corrections et les notes sont réutilisées par l’assistant.</p>
        </div>
        <nav className="tabs">
          {TABS.map((t) => (
            <a key={t.id} href={`#/console/${t.id}`} className={`tab ${tab === t.id ? 'active' : ''}`}>
              {t.label}
            </a>
          ))}
        </nav>
      </div>
      {tab === 'dashboard' && <Dashboard />}
      {tab === 'traces' && <Traces />}
      {tab === 'corrections' && <Corrections />}
      {tab === 'prompts' && <Prompts />}
      {tab === 'knowledge' && <Knowledge isAdmin={user.role === 'admin'} />}
      {tab === 'users' && <Users isAdmin={user.role === 'admin'} />}
    </main>
  );
}

const fmt = (v: number | null | undefined, d = 0) => (v == null ? '–' : Number(v).toFixed(d));

function Dashboard() {
  const [s, setS] = useState<any>(null);
  useEffect(() => {
    api('/admin/stats').then(setS).catch(() => {});
  }, []);
  if (!s) return <div className="card pad">Chargement…</div>;
  const up = s.votes.up ?? 0;
  const down = s.votes.down ?? 0;
  const sat = up + down ? Math.round((100 * up) / (up + down)) : null;
  const max = Math.max(1, ...s.daily.map((d: any) => d.n));
  const n = s.totals.n || 0;
  return (
    <div className="dash">
      <div className="kpis">
        <Kpi label="Questions (30 j)" value={n} />
        <Kpi label="Satisfaction utilisateurs" value={sat == null ? '–' : `${sat} %`} hint={`${up} 👍 · ${down} 👎`} />
        <Kpi label="Note équipe HLP" value={s.reviewer.n ? `${fmt(s.reviewer.avg, 1)} / 5` : '–'} hint={`${s.reviewer.n} revues`} />
        <Kpi label="Note auto (LLM juge)" value={s.auto.n ? `${fmt(s.auto.avg, 1)} / 5` : '–'} hint={`${s.auto.n} évaluations`} />
        <Kpi label="À revoir en priorité" value={s.toReview} hint="votes négatifs ou note auto < 3,5" accent={s.toReview > 0} link="#/console/traces" />
        <Kpi label="Latence moyenne" value={n ? `${fmt(s.totals.lat / 1000, 1)} s` : '–'} />
        <Kpi label="Recherche web utilisée" value={n ? `${Math.round((100 * (s.totals.web ?? 0)) / n)} %` : '–'} />
        <Kpi label="Corrections actives" value={s.corrections} hint={`${s.chunks} extraits indexés`} />
      </div>

      <div className="grid-2">
        <section className="card pad">
          <h3>Questions par jour</h3>
          {s.daily.length === 0 && <p className="muted small">Pas encore de données.</p>}
          <div className="bars">
            {s.daily.map((d: any) => (
              <div key={d.day} className="bar-col" title={`${d.day} : ${d.n}`}>
                <div className="bar" style={{ height: `${(100 * d.n) / max}%` }} />
                <span className="xsmall muted">{d.day.slice(5)}</span>
              </div>
            ))}
          </div>
        </section>
        <section className="card pad">
          <h3>Performance par version du prompt système</h3>
          <table className="table">
            <thead>
              <tr><th>Version</th><th>Réponses</th><th>Satisfaction</th><th>Équipe</th><th>Auto</th><th>Latence</th></tr>
            </thead>
            <tbody>
              {s.byPrompt.map((p: any) => (
                <tr key={p.id}>
                  <td>v{p.version} {p.active ? <span className="badge green">active</span> : null}</td>
                  <td>{p.n}</td>
                  <td>{p.satisfaction == null ? '–' : `${Math.round(p.satisfaction * 100)} %`}</td>
                  <td>{fmt(p.reviewer, 1)}</td>
                  <td>{fmt(p.auto, 1)}</td>
                  <td>{p.lat ? `${fmt(p.lat / 1000, 1)} s` : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="xsmall muted">Comparez les versions pour décider laquelle garder active.</p>
        </section>
      </div>
      <section className="card pad">
        <h3>Usagers (30 j)</h3>
        <p className="small">{s.users.members ?? 0} membres connectés · {s.users.guests ?? 0} invités · {fmt((s.totals.tin ?? 0) / 1000, 0)} k tokens entrée · {fmt((s.totals.tout ?? 0) / 1000, 0)} k tokens sortie · {s.totals.err ?? 0} erreurs</p>
      </section>
    </div>
  );
}

function Kpi({ label, value, hint, accent, link }: { label: string; value: React.ReactNode; hint?: string; accent?: boolean; link?: string }) {
  const body = (
    <>
      <div className="muted xsmall upper">{label}</div>
      <div className="kpi-value">{value}</div>
      {hint && <div className="muted xsmall">{hint}</div>}
    </>
  );
  return link ? (
    <a href={link} className={`card kpi ${accent ? 'accent' : ''}`}>{body}</a>
  ) : (
    <div className={`card kpi ${accent ? 'accent' : ''}`}>{body}</div>
  );
}
