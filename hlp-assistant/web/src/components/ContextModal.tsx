import { useState } from 'react';
import { api, type Profile, type User } from '../api';
import { Modal } from './LoginModal';

const SECTORS = ['HLP', 'SLSC (Shelter, Land and Site Coordination)', 'Shelter / Abris', 'CCCM', 'Protection', 'ICLA / Assistance juridique', 'Coordination de cluster', 'Bailleur', 'Autorité locale / gouvernement', 'Organisation communautaire', 'Recherche', 'Autre'];
const LEVELS = ['Débutant', 'Intermédiaire', 'Expert'];

export function ContextModal({ user, onClose, onSaved }: { user: User; onClose: () => void; onSaved: (u: User) => void }) {
  const [name, setName] = useState(user.displayName ?? '');
  const [p, setP] = useState<Profile>(user.profile ?? {});
  const [error, setError] = useState('');
  const set = (k: keyof Profile) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => setP({ ...p, [k]: e.target.value });

  const save = async () => {
    try {
      const r = await api<{ user: User }>('/me/profile', { method: 'PUT', body: { displayName: name, profile: p } });
      onSaved(r.user);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  return (
    <Modal title="Mon contexte" onClose={onClose} wide>
      <form className="form grid2" onSubmit={(e) => { e.preventDefault(); save(); }}>
        <p className="muted span2">
          Ces informations permettent à l’assistant d’adapter ses réponses à votre rôle et à votre terrain. Tout est facultatif.
        </p>
        <label className="span2">
          Nom ou pseudo
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Ex. Amina K." autoFocus />
        </label>
        <label>
          Fonction
          <input value={p.jobTitle ?? ''} onChange={set('jobTitle')} placeholder="Ex. HLP Officer, Shelter Coordinator" />
        </label>
        <label>
          Organisation
          <input value={p.organisation ?? ''} onChange={set('organisation')} placeholder="Ex. NRC, ONG locale, mairie…" />
        </label>
        <label>
          Secteur
          <select value={p.sector ?? ''} onChange={set('sector')}>
            <option value="">—</option>
            {SECTORS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        <label>
          Pays / opération
          <input value={p.country ?? ''} onChange={set('country')} placeholder="Ex. RDC — Nord-Kivu" />
        </label>
        <label>
          Niveau en HLP
          <select value={p.expertise ?? ''} onChange={set('expertise')}>
            <option value="">—</option>
            {LEVELS.map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        <label>
          Langue préférée
          <input value={p.language ?? ''} onChange={set('language')} placeholder="Français, English, العربية…" />
        </label>
        <label className="span2">
          Centres d’intérêt / enjeux actuels
          <textarea value={p.interests ?? ''} onChange={set('interests')} placeholder="Ex. expulsions dans les sites urbains, droits fonciers des femmes, retour…" />
        </label>
        {error && <div className="alert span2">{error}</div>}
        <div className="row end span2">
          <button type="button" className="btn ghost" onClick={onClose}>Plus tard</button>
          <button className="btn primary">Enregistrer</button>
        </div>
      </form>
    </Modal>
  );
}
