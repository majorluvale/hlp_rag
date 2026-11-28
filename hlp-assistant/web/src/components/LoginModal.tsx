import { useState, type ReactNode } from 'react';
import { api } from '../api';
import { useI18n } from '../i18n';
import { IconX } from './Icons';

export function Modal({ title, onClose, children, wide }: { title: string; onClose: () => void; children: ReactNode; wide?: boolean }) {
  const { t } = useI18n();
  return (
    <div className="modal-backdrop" onMouseDown={onClose}>
      <div className={`modal card ${wide ? 'wide' : ''}`} onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-modal>
        <div className="modal-head">
          <h2>{title}</h2>
          <button className="icon-btn ghost" onClick={onClose} aria-label={t('common.close')}>
            <IconX />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function LoginModal({ onClose, onLoggedIn }: { onClose: () => void; onLoggedIn: (needsName: boolean) => void }) {
  const { t } = useI18n();
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
      setNotice(r.delivered ? t('login.sent', { email }) : t('login.demo'));
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
    <Modal title={t('login.title')} onClose={onClose}>
      {step === 'email' ? (
        <form onSubmit={(e) => { e.preventDefault(); request(); }} className="form">
          <p className="muted">{t('login.intro')}</p>
          <label>
            {t('login.email')}
            <input type="email" autoFocus required value={email} onChange={(e) => setEmail(e.target.value)} placeholder={t('login.emailPlaceholder')} />
          </label>
          {error && <div className="alert">{error}</div>}
          <button className="btn primary block lg" disabled={loading}>{loading ? t('login.sending') : t('login.getCode')}</button>
          <p className="xsmall muted center">{t('login.guestHint')}</p>
        </form>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); verify(); }} className="form">
          <p className="muted">{notice}</p>
          {devCode && <div className="alert info">{t('login.devCode')} <strong>{devCode}</strong></div>}
          <label>
            {t('login.code')}
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
          <button className="btn primary block lg" disabled={loading || code.length !== 6}>{loading ? t('login.verifying') : t('login.submit')}</button>
          <div className="row between">
            <button type="button" className="btn ghost" onClick={() => setStep('email')}>{t('login.changeEmail')}</button>
            <button type="button" className="btn ghost" onClick={request} disabled={loading}>{t('login.resend')}</button>
          </div>
        </form>
      )}
    </Modal>
  );
}
