import { useCallback, useEffect, useState } from 'react';
import { api, initials, type Conversation, type User } from './api';
import { ChatPage } from './components/ChatPage';
import { LoginModal } from './components/LoginModal';
import { ContextModal } from './components/ContextModal';
import { Console } from './console/Console';
import { IconChart, IconHome, IconLogout, IconMenu, IconShield, IconUser } from './components/Icons';
import { ACRONYM, useI18n, type Lang } from './i18n';

export type Route = 'chat' | 'console';
const readRoute = (): Route => (location.hash.startsWith('#/console') ? 'console' : 'chat');
const isNarrow = () => window.matchMedia('(max-width: 760px)').matches;
const SIDEBAR_KEY = 'hlp.sidebar';
const initialSidebar = () => {
  if (isNarrow()) return false; // phones: the menu is a drawer, closed by default
  try {
    return localStorage.getItem(SIDEBAR_KEY) !== 'closed';
  } catch {
    return true;
  }
};

export function App() {
  const { t, lang, setLang } = useI18n();
  const [user, setUser] = useState<User | null>(null);
  const [route, setRoute] = useState<Route>(readRoute());
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loginOpen, setLoginOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(initialSidebar);
  const toggleSidebar = (open = !sidebarOpen) => {
    setSidebarOpen(open);
    if (!isNarrow()) {
      try {
        localStorage.setItem(SIDEBAR_KEY, open ? 'open' : 'closed');
      } catch {
        /* storage unavailable */
      }
    }
  };

  const refreshMe = useCallback(() => api<{ user: User }>('/me').then((r) => setUser(r.user)), []);
  const refreshConversations = useCallback(
    () => api<{ conversations: Conversation[] }>('/conversations').then((r) => setConversations(r.conversations)),
    [],
  );

  useEffect(() => {
    refreshMe().then(refreshConversations).catch(() => {});
    const onHash = () => setRoute(readRoute());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, [refreshMe, refreshConversations]);

  useEffect(() => {
    document.title = t('assistant.name');
  }, [t]);

  const isTeam = user?.role === 'reviewer' || user?.role === 'admin';
  const guestLabel = t('common.guest');

  const logout = async () => {
    await api('/auth/logout', { method: 'POST', body: {} });
    setMenuOpen(false);
    location.hash = '#/';
    await refreshMe();
    await refreshConversations();
  };

  return (
    <div className="app">
      <header className="nav">
        <div className="nav-inner">
          <div className="nav-left">
            {route === 'chat' && (
              <button className="icon-btn burger" onClick={() => toggleSidebar()} title={t('nav.toggleSidebar')} aria-label={t('nav.toggleSidebar')} aria-expanded={sidebarOpen}>
                <IconMenu />
              </button>
            )}
            <a className="brand" href="#/">
              <span className="brand-logo">{ACRONYM[lang]}</span>
              <span className="brand-text">
                <strong>{t('assistant.name')}</strong>
                <small>{t('brand.tagline')}</small>
              </span>
            </a>
          </div>
          <nav className="nav-items">
            <a className={`nav-item ${route === 'chat' ? 'active' : ''}`} href="#/">
              <IconHome />
              <span>{t('nav.home')}</span>
            </a>
            {isTeam && (
              <a className={`nav-item ${route === 'console' ? 'active' : ''}`} href="#/console">
                <IconChart />
                <span>{t('nav.observability')}</span>
              </a>
            )}
            <div className="nav-item lang-switch" role="group" aria-label={t('nav.language')}>
              <div className="seg mini">
                {(['en', 'fr'] as Lang[]).map((l) => (
                  <button key={l} className={lang === l ? 'on' : ''} onClick={() => setLang(l)} aria-pressed={lang === l}>
                    {l.toUpperCase()}
                  </button>
                ))}
              </div>
              <span>{t('nav.language')}</span>
            </div>
            <div className="nav-item me" onClick={() => setMenuOpen((o) => !o)} role="button" tabIndex={0}>
              <span className="avatar xs">{user && !user.isGuest ? initials(user, guestLabel) : <IconUser width={16} height={16} />}</span>
              <span>{t('nav.me')} ▾</span>
              {menuOpen && (
                <div className="menu" onClick={(e) => e.stopPropagation()}>
                  <div className="menu-head">
                    <span className="avatar md">{initials(user, guestLabel)}</span>
                    <div>
                      <strong>{user?.displayName || (user?.isGuest ? guestLabel : user?.email)}</strong>
                      <div className="muted small">{user?.isGuest ? t('menu.guestSession') : user?.email}</div>
                    </div>
                  </div>
                  <button className="btn outline block" onClick={() => { setContextOpen(true); setMenuOpen(false); }}>
                    {t('menu.myContext')}
                  </button>
                  {user?.isGuest ? (
                    <button className="btn primary block" onClick={() => { setLoginOpen(true); setMenuOpen(false); }}>
                      {t('menu.signIn')}
                    </button>
                  ) : (
                    <button className="menu-link" onClick={logout}>
                      <IconLogout width={16} height={16} /> {t('menu.signOut')}
                    </button>
                  )}
                  {isTeam && (
                    <div className="menu-note">
                      <IconShield width={14} height={14} /> {t('menu.team', { role: user?.role ?? '' })}
                    </div>
                  )}
                </div>
              )}
            </div>
          </nav>
        </div>
      </header>

      {route === 'console' && isTeam ? (
        <Console user={user!} />
      ) : (
        <ChatPage
          user={user}
          conversations={conversations}
          refreshConversations={refreshConversations}
          onLogin={() => setLoginOpen(true)}
          onEditContext={() => setContextOpen(true)}
          sidebarOpen={sidebarOpen}
          onCloseSidebarOnMobile={() => isNarrow() && toggleSidebar(false)}
        />
      )}

      {loginOpen && (
        <LoginModal
          onClose={() => setLoginOpen(false)}
          onLoggedIn={async (needsName) => {
            await refreshMe();
            await refreshConversations();
            setLoginOpen(false);
            if (needsName) setContextOpen(true);
          }}
        />
      )}
      {contextOpen && user && (
        <ContextModal
          user={user}
          onClose={() => setContextOpen(false)}
          onSaved={(u) => {
            setUser(u);
            setContextOpen(false);
          }}
        />
      )}
    </div>
  );
}
