import { useState, type ChangeEvent } from 'react';
import { api, type Profile, type User } from '../api';
import { useI18n } from '../i18n';
import { Modal } from './LoginModal';

export function ContextModal({ user, onClose, onSaved }: { user: User; onClose: () => void; onSaved: (u: User) => void }) {
  const { t } = useI18n();
  const sectors = t('ctx.sectors').split('|');
  const levels = t('ctx.levels').split('|');
  const [name, setName] = useState(user.displayName ?? '');
  const [p, setP] = useState<Profile>(user.profile ?? {});
  const [error, setError] = useState('');
  const set = (k: keyof Profile) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => setP({ ...p, [k]: e.target.value });

  // keep a previously saved value selectable even if it was saved in the other language
  const withCurrent = (list: string[], v?: string) => (v && !list.includes(v) ? [...list, v] : list);

  const save = async () => {
    try {
      const r = await api<{ user: User }>('/me/profile', { method: 'PUT', body: { displayName: name, profile: p } });
      onSaved(r.user);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <Modal title={t('ctx.title')} onClose={onClose} wide>
      <form className="form grid2" onSubmit={(e) => { e.preventDefault(); save(); }}>
        <p className="muted span2">{t('ctx.intro')}</p>
        <label className="span2">
          {t('ctx.name')}
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder={t('ctx.namePh')} autoFocus />
        </label>
        <label>
          {t('ctx.job')}
          <input value={p.jobTitle ?? ''} onChange={set('jobTitle')} placeholder={t('ctx.jobPh')} />
        </label>
        <label>
          {t('ctx.org')}
          <input value={p.organisation ?? ''} onChange={set('organisation')} placeholder={t('ctx.orgPh')} />
        </label>
        <label>
          {t('ctx.sector')}
          <select value={p.sector ?? ''} onChange={set('sector')}>
            <option value="">—</option>
            {withCurrent(sectors, p.sector).map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        <label>
          {t('ctx.country')}
          <input value={p.country ?? ''} onChange={set('country')} placeholder={t('ctx.countryPh')} />
        </label>
        <label>
          {t('ctx.level')}
          <select value={p.expertise ?? ''} onChange={set('expertise')}>
            <option value="">—</option>
            {withCurrent(levels, p.expertise).map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        <label>
          {t('ctx.language')}
          <input value={p.language ?? ''} onChange={set('language')} placeholder={t('ctx.languagePh')} />
        </label>
        <label className="span2">
          {t('ctx.interests')}
          <textarea value={p.interests ?? ''} onChange={set('interests')} placeholder={t('ctx.interestsPh')} />
        </label>
        {error && <div className="alert span2">{error}</div>}
        <div className="row end span2">
          <button type="button" className="btn ghost" onClick={onClose}>{t('common.later')}</button>
          <button className="btn primary">{t('common.save')}</button>
        </div>
      </form>
    </Modal>
  );
}
