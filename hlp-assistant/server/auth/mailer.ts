import nodemailer from 'nodemailer';
import { config } from '../config';
import { msg, type Lang } from '../i18n';

const transporter = config.smtp.host
  ? nodemailer.createTransport({
      host: config.smtp.host,
      port: config.smtp.port,
      secure: config.smtp.port === 465,
      auth: config.smtp.user ? { user: config.smtp.user, pass: config.smtp.pass } : undefined,
    })
  : null;

export const mailConfigured = () => Boolean(transporter);

export async function sendOtpEmail(email: string, code: string, lang: Lang = 'en'): Promise<{ delivered: boolean }> {
  if (!transporter) {
    console.log(`\n[OTP — SMTP not configured] ${email} → ${code}\n`);
    return { delivered: false };
  }
  const brand = msg(lang, 'mail.brand');
  await transporter.sendMail({
    from: config.smtp.from,
    to: email,
    subject: msg(lang, 'mail.subject', { code }),
    text: `${msg(lang, 'mail.hello')}\n\n${msg(lang, 'mail.code')} ${code}\n\n${msg(lang, 'mail.expires')}`,
    html: `<div style="font-family:Segoe UI,Arial,sans-serif;max-width:480px;margin:auto;padding:24px;border:1px solid #e0dfdc;border-radius:8px">
      <div style="font-weight:700;color:#0a66c2;font-size:20px">${brand}</div>
      <p>${msg(lang, 'mail.hello')}</p><p>${msg(lang, 'mail.code')}</p>
      <div style="font-size:32px;letter-spacing:8px;font-weight:700;background:#f4f2ee;padding:12px;text-align:center;border-radius:6px">${code}</div>
      <p style="color:#666;font-size:13px">${msg(lang, 'mail.expires')}</p></div>`,
  });
  return { delivered: true };
}
