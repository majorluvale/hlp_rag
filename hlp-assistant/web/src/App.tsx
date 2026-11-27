import { useCallback, useEffect, useState } from 'react';
import { api, initials, type Conversation, type User } from './api';
import { ChatPage } from './components/ChatPage';
import { LoginModal } from './components/LoginModal';
import { ContextModal } from './components/ContextModal';
import { Console } from './console/Console';
import { IconChart, IconHome, IconLogout, IconShield, IconUser } from './components/Icons';

export type Route = 'chat' | 'console';
const readRoute = (): Route => (location.hash.startsWith('#/console') ? 'console' : 'chat');

export function App() {
  const [user, setUser] = useState<User | null>(null);
  const [route, setRoute] = useState<Route>(readRoute());
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loginOpen, setLoginOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

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

  const isTeam = user?.role === 'reviewer' || user?.role === 'admin';

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
          <a className="brand" href="#/">
            <span className="brand-logo">HLP</span>
            <span className="brand-text">
              <strong>HLP Assistant</strong>
              <small>Housing · Land · Property</small>
            </span>
          </a>
          <nav className="nav-items">
            <a className={`nav-item ${route === 'chat' ? 'active' : ''}`} href="#/">
              <IconHome />
              <span>Accueil</span>
            </a>
            {isTeam && (
              <a className={`nav-item ${route === 'console' ? 'active' : ''}`} href="#/console">
                <IconChart />
                <span>Observabilité</span>
              </a>
            )}
            <div className="nav-item me" onClick={() => setMenuOpen((o) => !o)} role="button" tabIndex={0}>
              <span className="avatar xs">{user && !user.isGuest ? initials(user) : <IconUser width={16} height={16} />}</span>
              <span>Moi ▾</span>
              {menuOpen && (
                <div className="menu" onClick={(e) => e.stopPropagation()}>
                  <div className="menu-head">
                    <span className="avatar md">{initials(user)}</span>
                    <div>
                      <strong>{user?.displayName || (user?.isGuest ? 'Invité' : user?.email)}</strong>
                      <div className="muted small">{user?.isGuest ? 'Session invitée' : user?.email}</div>
                    </div>
                  </div>
                  <button className="btn outline block" onClick={() => { setContextOpen(true); setMenuOpen(false); }}>
                    Mon contexte
                  </button>
                  {user?.isGuest ? (
                    <button className="btn primary block" onClick={() => { setLoginOpen(true); setMenuOpen(false); }}>
                      Se connecter
                    </button>
                  ) : (
                    <button className="menu-link" onClick={logout}>
                      <IconLogout width={16} height={16} /> Se déconnecter
                    </button>
                  )}
                  {isTeam && (
                    <div className="menu-note">
                      <IconShield width={14} height={14} /> Membre de l’équipe HLP ({user?.role})
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
