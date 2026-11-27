import crypto from 'node:crypto';
import type { NextFunction, Request, Response } from 'express';
import { config } from '../config';
import { batch, get, now, q, run, uid, type UserRow } from '../db';
import { msg, type Lang } from '../i18n';
import { sendOtpEmail } from './mailer';

const COOKIE = 'hlp_sid';
const SESSION_TTL = 60 * 24 * 3600 * 1000; // 60 days
const OTP_TTL = 10 * 60 * 1000;
const OTP_RESEND_DELAY = 45 * 1000;
const OTP_MAX_ATTEMPTS = 5;

declare global {
  // eslint-disable-next-line @typescript-eslint/no-namespace
  namespace Express {
    interface Request {
      user: UserRow;
    }
  }
}

const sha256 = (s: string) => crypto.createHash('sha256').update(s).digest('hex');
const hmac = (s: string) => crypto.createHmac('sha256', config.sessionSecret).update(s).digest('hex');

function setSessionCookie(req: Request, res: Response, token: string) {
  res.cookie(COOKIE, token, {
    httpOnly: true,
    sameSite: 'lax',
    secure: req.secure,
    maxAge: SESSION_TTL,
    path: '/',
  });
}

async function createSession(userId: string): Promise<string> {
  const token = crypto.randomBytes(32).toString('base64url');
  await run('INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)', sha256(token), userId, now(), now() + SESSION_TTL);
  return token;
}

async function createGuest(): Promise<UserRow> {
  const id = uid('u_');
  const t = now();
  await run("INSERT INTO users (id, role, is_guest, created_at, last_seen_at) VALUES (?, 'guest', 1, ?, ?)", id, t, t);
  return (await get<UserRow>('SELECT * FROM users WHERE id = ?', id))!;
}

/** Every visitor gets an identity: a guest by default (no login needed), a member after OTP. */
export async function attachUser(req: Request, res: Response, next: NextFunction) {
  const token = req.cookies?.[COOKIE] as string | undefined;
  if (token) {
    const row = await get<UserRow & { expires_at: number }>(
      'SELECT u.*, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ?',
      sha256(token),
    );
    if (row && row.expires_at > now()) {
      req.user = row;
      if (now() - row.last_seen_at > 60000) await run('UPDATE users SET last_seen_at = ? WHERE id = ?', now(), row.id);
      return next();
    }
  }
  // Only API calls create guests (avoids creating users for static asset requests)
  if (!req.path.startsWith('/api/')) return next();
  const guest = await createGuest();
  setSessionCookie(req, res, await createSession(guest.id));
  req.user = guest;
  next();
}

export function requireRole(...roles: UserRow['role'][]) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user || !roles.includes(req.user.role)) {
      const lang: Lang = String(req.headers['x-lang'] ?? '').startsWith('fr') ? 'fr' : 'en';
      return res.status(403).json({ error: msg(lang, 'err.team') });
    }
    next();
  };
}

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export async function requestOtp(emailRaw: string, isLocalRequest = false, lang: Lang = 'en') {
  const email = String(emailRaw ?? '').trim().toLowerCase();
  if (!EMAIL_RE.test(email) || email.length > 200) return { status: 400, body: { error: msg(lang, 'err.email') } };
  const last = await get<{ created_at: number }>('SELECT created_at FROM otp_codes WHERE email = ? ORDER BY created_at DESC LIMIT 1', email);
  if (last && now() - last.created_at < OTP_RESEND_DELAY) {
    return { status: 429, body: { error: msg(lang, 'err.otpWait') } };
  }
  const code = String(crypto.randomInt(0, 1_000_000)).padStart(6, '0');
  await batch([
    q('UPDATE otp_codes SET consumed = 1 WHERE email = ? AND consumed = 0', email),
    q('INSERT INTO otp_codes (email, code_hash, expires_at, created_at) VALUES (?, ?, ?, ?)', email, hmac(email + ':' + code), now() + OTP_TTL, now()),
  ]);
  const { delivered } = await sendOtpEmail(email, code, lang);
  return {
    status: 200,
    body: {
      ok: true,
      delivered,
      // the demo code is only ever returned to a browser on localhost, never through a tunnel or online
      ...(config.devShowOtp && !delivered && isLocalRequest ? { devCode: code } : {}),
    },
  };
}

export async function verifyOtp(req: Request, res: Response, emailRaw: string, codeRaw: string, lang: Lang = 'en') {
  const email = String(emailRaw ?? '').trim().toLowerCase();
  const code = String(codeRaw ?? '').replace(/\D/g, '');
  const otp = await get<{ id: number; code_hash: string; attempts: number; expires_at: number }>(
    'SELECT * FROM otp_codes WHERE email = ? AND consumed = 0 ORDER BY created_at DESC LIMIT 1',
    email,
  );
  if (!otp || otp.expires_at < now()) return { status: 400, body: { error: msg(lang, 'err.otpExpired') } };
  if (otp.attempts >= OTP_MAX_ATTEMPTS) return { status: 429, body: { error: msg(lang, 'err.otpAttempts') } };
  const ok = code.length === 6 && crypto.timingSafeEqual(Buffer.from(otp.code_hash), Buffer.from(hmac(email + ':' + code)));
  if (!ok) {
    await run('UPDATE otp_codes SET attempts = attempts + 1 WHERE id = ?', otp.id);
    return { status: 400, body: { error: msg(lang, 'err.otpWrong') } };
  }
  await run('UPDATE otp_codes SET consumed = 1 WHERE id = ?', otp.id);

  const guest = req.user?.is_guest ? req.user : null;
  const role = config.adminEmails.includes(email) ? 'admin' : 'user';
  let user = await get<UserRow>('SELECT * FROM users WHERE email = ?', email);
  if (!user) {
    if (guest) {
      // promote the guest account: keeps its conversations and context
      await run('UPDATE users SET email = ?, is_guest = 0, role = ? WHERE id = ?', email, role, guest.id);
      user = (await get<UserRow>('SELECT * FROM users WHERE id = ?', guest.id))!;
    } else {
      const id = uid('u_');
      await run('INSERT INTO users (id, email, role, is_guest, created_at, last_seen_at) VALUES (?, ?, ?, 0, ?, ?)', id, email, role, now(), now());
      user = (await get<UserRow>('SELECT * FROM users WHERE id = ?', id))!;
    }
  } else {
    const stmts = [];
    if (role === 'admin' && user.role !== 'admin') stmts.push(q("UPDATE users SET role = 'admin' WHERE id = ?", user.id));
    if (guest && guest.id !== user.id) {
      // merge the guest history into the existing account
      stmts.push(
        q('UPDATE conversations SET user_id = ? WHERE user_id = ?', user.id, guest.id),
        q('UPDATE traces SET user_id = ? WHERE user_id = ?', user.id, guest.id),
        q('UPDATE feedback SET user_id = ? WHERE user_id = ?', user.id, guest.id),
        q('DELETE FROM sessions WHERE user_id = ?', guest.id),
        q('DELETE FROM users WHERE id = ?', guest.id),
      );
      if (user.profile_json === '{}' && guest.profile_json !== '{}') stmts.push(q('UPDATE users SET profile_json = ? WHERE id = ?', guest.profile_json, user.id));
    }
    await batch(stmts);
    user = (await get<UserRow>('SELECT * FROM users WHERE id = ?', user.id))!;
  }

  const old = req.cookies?.[COOKIE] as string | undefined;
  if (old) await run('DELETE FROM sessions WHERE token_hash = ?', sha256(old));
  setSessionCookie(req, res, await createSession(user.id));
  return { status: 200, body: { user: publicUser(user), needsName: !user.display_name } };
}

export async function logout(req: Request, res: Response) {
  const token = req.cookies?.[COOKIE] as string | undefined;
  if (token) await run('DELETE FROM sessions WHERE token_hash = ?', sha256(token));
  res.clearCookie(COOKIE, { path: '/' });
}

export function publicUser(u: UserRow) {
  return {
    id: u.id,
    email: u.email,
    displayName: u.display_name,
    role: u.role,
    isGuest: Boolean(u.is_guest),
    profile: JSON.parse(u.profile_json || '{}'),
  };
}
