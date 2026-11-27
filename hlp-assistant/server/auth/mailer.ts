import nodemailer from 'nodemailer';
import { config } from '../config';

const transporter = config.smtp.host
  ? nodemailer.createTransport({
      host: config.smtp.host,
      port: config.smtp.port,
      secure: config.smtp.port === 465,
      auth: config.smtp.user ? { user: config.smtp.user, pass: config.smtp.pass } : undefined,
    })
  : null;

export const mailConfigured = () => Boolean(transporter);

export async function sendOtpEmail(email: string, code: string): Promise<{ delivered: boolean }> {
  if (!transporter) {
    console.log(`\n[OTP — SMTP non configuré] ${email} → ${code}\n`);
    return { delivered: false };
  }
  await transporter.sendMail({
    from: config.smtp.from,
    to: email,
    subject: `${code} — votre code de connexion HLP Assistant`,
    text: `Bonjour,\n\nVotre code de connexion HLP Assistant est : ${code}\nIl expire dans 10 minutes.\n\nSi vous n'êtes pas à l'origine de cette demande, ignorez ce message.`,
    html: `<div style="font-family:Segoe UI,Arial,sans-serif;max-width:480px;margin:auto;padding:24px;border:1px solid #e0dfdc;border-radius:8px">
      <div style="font-weight:700;color:#0a66c2;font-size:20px">HLP Assistant</div>
      <p>Bonjour,</p><p>Votre code de connexion :</p>
      <div style="font-size:32px;letter-spacing:8px;font-weight:700;background:#f4f2ee;padding:12px;text-align:center;border-radius:6px">${code}</div>
      <p style="color:#666;font-size:13px">Il expire dans 10 minutes. Si vous n'êtes pas à l'origine de cette demande, ignorez ce message.</p></div>`,
  });
  return { delivered: true };
}
