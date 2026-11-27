import { useEffect, useState, type ReactNode } from 'react';
import { api, type User } from '../api';
import { useI18n } from '../i18n';
import { Traces } from './Traces';
import { Prompts } from './Prompts';
import { Corrections, Knowledge, Users } from './Others';
import { Settings } from './Settings';

type Tab = 'dashboard' | 'traces' | 'corrections' | 'prompts' | 'knowledge' | 'users' | 'settings';
const TABS: Tab[] = ['dashboard', 'traces', 'corrections', 'prompts', 'knowledge', 'users', 'settings'];

const readTab = (): Tab => {
  const t = location.hash.split('/')[2] as Tab;
  return TABS.includes(t) ? t : 'dashboard';
};

export function Console({ user }: { user: User }) {
  const { t } = useI18n();
  const [tab, setTab] = useState<Tab>(readTab());
  useEffect(() => {
    const on = () => setTab(readTab());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  const isAdmin = user.role === 'admin';

  return (
    <main className="console">
      <div className="card console-head">
        <div>
          <h1>{t('console.title')}</h1>
          <p className="muted small">{t('console.subtitle')}</p>
        </div>
        <nav className="tabs">
          {TABS.filter((id) => id !== 'settings' || isAdmin).map((id) => (
            <a key={id} href={`#/console/${id}`} className={`tab ${tab === id ? 'active' : ''}`}>
              {t(`tab.${id}`)}
            </a>
          ))}
        </nav>
      </div>
      {tab === 'dashboard' && <Dashboard />}
      {tab === 'traces' && <Traces />}
      {tab === 'corrections' && <Corrections />}
      {tab === 'prompts' && <Prompts />}
      {tab === 'knowledge' && <Knowledge isAdmin={isAdmin} />}
      {tab === 'users' && <Users isAdmin={isAdmin} />}
      {tab === 'settings' && isAdmin && <Settings />}
    </main>
  );
}

const fmt = (v: number | null | undefined, d = 0) => (v == null ? '–' : Number(v).toFixed(d));

function Dashboard() {
  const { t } = useI18n();
  const [s, setS] = useState<any>(null);
  useEffect(() => {
    api('/admin/stats').then(setS).catch(() => {});
  }, []);
  if (!s) return <div className="card pad">{t('common.loading')}</div>;
  const up = s.votes.up ?? 0;
  const down = s.votes.down ?? 0;
  const sat = up + down ? Math.round((100 * up) / (up + down)) : null;
  const max = Math.max(1, ...s.daily.map((d: any) => d.n));
  const n = s.totals.n || 0;
  return (
    <div className="dash">
      <div className="kpis">
        <Kpi label={t('dash.questions')} value={n} />
        <Kpi label={t('dash.satisfaction')} value={sat == null ? '–' : `${sat} %`} hint={`${up} 👍 · ${down} 👎`} />
        <Kpi label={t('dash.teamScore')} value={s.reviewer.n ? `${fmt(s.reviewer.avg, 1)} / 5` : '–'} hint={t('dash.reviews', { n: s.reviewer.n })} />
        <Kpi label={t('dash.autoScore')} value={s.auto.n ? `${fmt(s.auto.avg, 1)} / 5` : '–'} hint={t('dash.evals', { n: s.auto.n })} />
        <Kpi label={t('dash.toReview')} value={s.toReview} hint={t('dash.toReviewHint')} accent={s.toReview > 0} link="#/console/traces" />
        <Kpi label={t('dash.latency')} value={n ? `${fmt(s.totals.lat / 1000, 1)} s` : '–'} />
        <Kpi label={t('dash.webUse')} value={n ? `${Math.round((100 * (s.totals.web ?? 0)) / n)} %` : '–'} />
        <Kpi label={t('dash.activeCorrections')} value={s.corrections} hint={t('dash.chunks', { n: s.chunks })} />
      </div>

      <div className="grid-2">
        <section className="card pad">
          <h3>{t('dash.perDay')}</h3>
          {s.daily.length === 0 && <p className="muted small">{t('dash.noData')}</p>}
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
          <h3>{t('dash.byPrompt')}</h3>
          <table className="table">
            <thead>
              <tr><th>{t('dash.version')}</th><th>{t('dash.answers')}</th><th>{t('dash.satisfaction')}</th><th>{t('dash.team')}</th><th>{t('dash.auto')}</th><th>{t('dash.latency')}</th></tr>
            </thead>
            <tbody>
              {s.byPrompt.map((p: any) => (
                <tr key={p.id}>
                  <td>v{p.version} {p.active ? <span className="badge green">{t('dash.active')}</span> : null}</td>
                  <td>{p.n}</td>
                  <td>{p.satisfaction == null ? '–' : `${Math.round(p.satisfaction * 100)} %`}</td>
                  <td>{fmt(p.reviewer, 1)}</td>
                  <td>{fmt(p.auto, 1)}</td>
                  <td>{p.lat ? `${fmt(p.lat / 1000, 1)} s` : '–'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="xsmall muted">{t('dash.compare')}</p>
        </section>
      </div>
      <section className="card pad">
        <h3>{t('dash.users')}</h3>
        <p className="small">
          {t('dash.usersLine', {
            members: s.users.members ?? 0,
            guests: s.users.guests ?? 0,
            tin: fmt((s.totals.tin ?? 0) / 1000, 0),
            tout: fmt((s.totals.tout ?? 0) / 1000, 0),
            err: s.totals.err ?? 0,
          })}
        </p>
      </section>
    </div>
  );
}

function Kpi({ label, value, hint, accent, link }: { label: string; value: ReactNode; hint?: string; accent?: boolean; link?: string }) {
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
