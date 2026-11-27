import { useState } from 'react';
import { api } from '../api';
import { IconX } from './Icons';

export function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children: React.ReactNode; wide?: boolean }) {
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className={`modal card ${wide ? 'wide' : ''}`} onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-modal>
        <div className="modal-head">
          <h2>{title}</h2>
          <button className="icon-btn ghost" onClick={onClose} aria-label="Fermer">
            <IconX />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function LoginModal({ onClose, onLoggedIn }: { onClose: () => void; onLoggedIn: (needsName: boolean) => void }) {
  const [step, setStep] = useState<'email' | 'code'>('email');
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [devCode, setDevCode] = useState<string | null>(null);
  const [notice, setNotice] = useState('');

  const request = async () => {
    setError('');
    setLoading(true);
    try {
      const r = await api<{ delivered: boolean; devCode?: string }>('/auth/request-otp', { body: { email } });
      setDevCode(r.devCode ?? null);
      setNotice(r.delivered ? `Un code à 6 chiffres a été envoyé à ${email}.` : 'Mode démo : l’email n’est pas configuré, le code est affiché dans la console du serveur.');
      setStep('code');
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const verify = async () => {
    setError('');
    setLoading(true);
    try {
      const r = await api<{ needsName: boolean }>('/auth/verify-otp', { body: { email, code } });
      onLoggedIn(r.needsName);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal title="Se connecter" onClose={onClose}>
      {step === 'email' ? (
        <form onSubmit={(e) => { e.preventDefault(); request(); }} className="form">
          <p className="muted">Pas de mot de passe : saisissez votre email, nous vous envoyons un code de connexion.</p>
          <label>
            Adresse email
            <input type="email" autoFocus required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="prenom.nom@nrc.no" />
          </label>
          {error && <div className="alert">{error}</div>}
          <button className="btn primary block lg" disabled={loading}>{loading ? 'Envoi…' : 'Recevoir un code'}</button>
          <p className="xsmall muted center">Vous pouvez aussi continuer en invité, sans connexion.</p>
        </form>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); verify(); }} className="form">
          <p className="muted">{notice}</p>
          {devCode && <div className="alert info">Code (mode développement) : <strong>{devCode}</strong></div>}
          <label>
            Code de vérification
            <input
              className="otp"
              inputMode="numeric"
              autoComplete="one-time-code"
              autoFocus
              maxLength={6}
              value={code}
              onChange={(e) => setCode(e.target.value.replace(/\D/g, ''))}
              placeholder="••••••"
            />
          </label>
          {error && <div className="alert">{error}</div>}
          <button className="btn primary block lg" disabled={loading || code.length !== 6}>{loading ? 'Vérification…' : 'Se connecter'}</button>
          <div className="row between">
            <button type="button" className="btn ghost" onClick={() => setStep('email')}>Changer d’email</button>
            <button type="button" className="btn ghost" onClick={request} disabled={loading}>Renvoyer le code</button>
          </div>
        </form>
      )}
    </Modal>
  );
}
