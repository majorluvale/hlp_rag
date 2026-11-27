import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import cookieParser from 'cookie-parser';
import { config } from './config';
import { get } from './db';
import { seedPrompts } from './ai/prompts';
import { llmConfigured } from './ai/llm';
import { attachUser } from './auth/auth';
import { mailConfigured } from './auth/mailer';
import { index, loadIndex } from './rag/hybridIndex';
import { ingestAll } from './rag/ingest';
import { chatRouter } from './routes/chat';
import { adminRouter } from './routes/admin';

seedPrompts();
const chunks = loadIndex();

const app = express();
app.set('trust proxy', true); // behind Cloudflare Tunnel / reverse proxy
app.disable('x-powered-by');
app.use(express.json({ limit: '1mb' }));
app.use(cookieParser());
app.use((_req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});
app.use(attachUser);

app.get('/api/health', (_req, res) => res.json({ ok: true, chunks: index.size, llm: llmConfigured(), mail: mailConfigured() }));
app.use('/api/admin', adminRouter);
app.use('/api', chatRouter);
app.use('/api', (_req, res) => res.status(404).json({ error: 'Route inconnue' }));

const dist = path.resolve('dist');
if (fs.existsSync(dist)) {
  app.use(express.static(dist, { index: false, maxAge: '1h' }));
  app.use((req, res, next) => {
    if (req.method !== 'GET') return next();
    res.sendFile(path.join(dist, 'index.html'));
  });
}

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error('[server]', err);
  res.status(500).json({ error: 'Erreur interne' });
});

app.listen(config.port, () => {
  console.log(`\nHLP Assistant → http://localhost:${config.port}`);
  console.log(`  base de connaissances : ${chunks} extraits indexés`);
  console.log(`  LLM : ${config.llm.model} @ ${config.llm.baseURL} ${llmConfigured() ? '' : '⚠ LLM_API_KEY manquante'}`);
  console.log(`  email OTP : ${mailConfigured() ? 'SMTP configuré' : 'mode dev (codes affichés dans la console)'}`);
  console.log(`  sites web : ${config.web.domains.join(', ')}\n`);
  const docCount = get<{ n: number }>("SELECT COUNT(*) n FROM documents WHERE source_type = 'doc'")!.n;
  if (config.autoIngest && docCount === 0) {
    console.log('[ingest] base vide : indexation automatique des documents en arrière-plan…');
    void ingestAll();
  }
});
