import type { Request } from 'express';

/** Server-side messages shown to users, in the interface language (English by default). */
export type Lang = 'en' | 'fr';

export function langOf(req: Request): Lang {
  const v = String(req.body?.lang ?? req.headers['x-lang'] ?? '').toLowerCase();
  return v.startsWith('fr') ? 'fr' : 'en';
}

const M = {
  'status.searchKb': { en: 'Searching the HLP knowledge base…', fr: 'Recherche dans la base de connaissances LTP…' },
  'status.writing': { en: 'Writing the answer…', fr: 'Rédaction de la réponse…' },
  'status.analysing': { en: 'Analysing the new sources…', fr: 'Analyse des nouvelles sources…' },
  'status.deepSearch': { en: 'Deeper search: “{q}”', fr: 'Recherche approfondie : « {q} »' },
  'status.webSearch': { en: 'Searching online{site}: “{q}”', fr: 'Recherche en ligne{site} : « {q} »' },
  'status.webSite': { en: ' on {s}', fr: ' sur {s}' },
  'status.reading': { en: 'Reading {h}…', fr: 'Lecture de {h}…' },
  'answer.fallback': {
    en: 'Sorry, I could not produce an answer. Could you rephrase your question?',
    fr: 'Désolé, je n’ai pas pu formuler de réponse. Pouvez-vous reformuler votre question ?',
  },
  'source.validated': { en: 'Answer validated by the HLP team', fr: 'Réponse validée par l’équipe LTP' },
  'err.noKey': {
    en: 'No LLM API key is configured. An administrator must add it in Observability › Settings.',
    fr: 'Aucune clé API LLM configurée. Un administrateur doit la renseigner dans Observabilité › Paramètres.',
  },
  'err.unavailable': {
    en: 'The service is temporarily unavailable. Please try again in a moment.',
    fr: 'Le service est momentanément indisponible. Réessayez dans un instant.',
  },
  'err.empty': { en: 'Empty message', fr: 'Message vide' },
  'err.tooLong': { en: 'Message too long (4000 characters max.)', fr: 'Message trop long (4000 caractères max.)' },
  'err.rate': {
    en: 'Question limit reached for this hour. Sign in for a higher limit.',
    fr: 'Limite de questions atteinte pour cette heure. Connectez-vous pour une limite plus élevée.',
  },
  'err.team': { en: 'Restricted to the HLP team', fr: 'Accès réservé à l’équipe LTP' },
  'err.email': { en: 'Invalid email address', fr: 'Adresse email invalide' },
  'err.otpWait': { en: 'A code was just sent. Please wait a few seconds.', fr: 'Un code vient d’être envoyé. Patientez quelques secondes.' },
  'err.otpExpired': { en: 'Code expired. Request a new code.', fr: 'Code expiré. Demandez un nouveau code.' },
  'err.otpAttempts': { en: 'Too many attempts. Request a new code.', fr: 'Trop de tentatives. Demandez un nouveau code.' },
  'err.otpWrong': { en: 'Incorrect code', fr: 'Code incorrect' },
  'err.mail': { en: 'Could not send the email. Check the SMTP configuration.', fr: "Impossible d'envoyer l'email. Vérifiez la configuration SMTP." },
  'err.notFound': { en: 'Not found', fr: 'Introuvable' },
  'mail.subject': { en: '{code} — your HLP Assistant sign-in code', fr: '{code} — votre code de connexion Assistant LTP' },
  'mail.hello': { en: 'Hello,', fr: 'Bonjour,' },
  'mail.code': { en: 'Your sign-in code:', fr: 'Votre code de connexion :' },
  'mail.expires': {
    en: 'It expires in 10 minutes. If you did not request it, you can ignore this message.',
    fr: 'Il expire dans 10 minutes. Si vous n’êtes pas à l’origine de cette demande, ignorez ce message.',
  },
  'mail.brand': { en: 'HLP Assistant', fr: 'Assistant LTP' },
} as const;

export type MsgKey = keyof typeof M;

export function msg(lang: Lang, key: MsgKey, vars?: Record<string, string>) {
  let s: string = M[key][lang] ?? M[key].en;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(v);
  return s;
}
