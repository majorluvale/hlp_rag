import fs from 'node:fs';
import path from 'node:path';
import express from 'express';
import cookieParser from 'cookie-parser';
import { config } from './config';
import { initDb } from './db';
import { seedPrompts } from './ai/prompts';
import { getLlmConfig, getSettings, setSettings } from './settings';
import { attachUser } from './auth/auth';
import { mailConfigured } from './auth/mailer';
import { embeddingInfo } from './rag/embeddings';
import { index, loadIndex } from './rag/hybridIndex';
import { chatRouter } from './routes/chat';
import { adminRouter } from './routes/admin';

/** Initialises storage, default prompts and the in-memory index (once per process / serverless instance). */
export const ready = (async () => {
  await initDb();
  await seedPrompts();
  const n = await loadIndex();
  const s = await getSettings();
  const indexEmbedding = s.get('index.embedding');
  if (n > 0 && !indexEmbedding) {
    // index built before this setting existed: it was built with the local e5 model
    await setSettings({ 'index.embedding': 'multilingual-e5-small:384' }, 'migration');
  } else if (indexEmbedding && !indexEmbedding.startsWith(embeddingInfo.family + ':')) {
    console.warn(
      `[index] ⚠ l'index a été construit avec ${indexEmbedding} mais le fournisseur actuel est ${embeddingInfo.model}. ` +
        'Ré-indexez (npm run ingest) ou utilisez un modèle de la même famille.',
    );
  }
})();
ready.catch(() => {}); // awaited (and reported) by the entry points

export const app = express();
app.set('trust proxy', true); // behind Vercel / Cloudflare Tunnel / reverse proxy
app.disable('x-powered-by');
app.use(express.json({ limit: '1mb' }));
app.use(cookieParser());
app.use((_req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
  next();
});

app.get('/api/health', async (_req, res) => {
  const llm = await getLlmConfig();
  res.json({ ok: true, chunks: index.size, llm: Boolean(llm.apiKey), model: llm.model, mail: mailConfigured(), embedding: embeddingInfo.provider });
});
app.use(attachUser);
app.use('/api/admin', adminRouter);
app.use('/api', chatRouter);
app.use('/api', (_req, res) => res.status(404).json({ error: 'Route inconnue' }));

// Local / Docker: the API also serves the built interface. On Vercel the CDN serves it.
const dist = path.resolve('dist');
if (!config.isVercel && fs.existsSync(dist)) {
  app.use(express.static(dist, { index: false, maxAge: '1h' }));
  app.use((req, res, next) => {
    if (req.method !== 'GET') return next();
    res.sendFile(path.join(dist, 'index.html'));
  });
}

app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error('[server]', err);
  if (!res.headersSent) res.status(500).json({ error: 'Erreur interne' });
});
