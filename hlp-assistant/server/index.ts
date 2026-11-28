import { config } from './config';
import { dbKind, get } from './db';
import { app, ready } from './app';
import { getLlmConfig } from './settings';
import { mailConfigured } from './auth/mailer';
import { embeddingInfo } from './rag/embeddings';
import { index } from './rag/hybridIndex';
import { ingestAll } from './rag/ingest';

// Long-running server (local PC, Docker, VM). On Vercel, server/vercel.ts is used instead.
await ready;
const server = app.listen(config.port, async () => {
  const llm = await getLlmConfig();
  console.log(`\nHLP Assistant → http://localhost:${config.port}`);
  console.log(`  base de données : ${dbKind() === 'turso' ? 'Turso ' + config.db.url : 'SQLite local'}`);
  console.log(`  base de connaissances : ${index.size} extraits indexés · embeddings ${embeddingInfo.provider} (${embeddingInfo.model})`);
  console.log(`  LLM : ${llm.model} @ ${llm.baseURL} (${llm.source === 'console' ? 'configuré dans la console' : '.env'}) ${llm.apiKey ? '' : '⚠ clé API manquante → Observabilité › Paramètres'}`);
  console.log(`  email OTP : ${mailConfigured() ? 'SMTP configuré' : 'mode dev (codes affichés dans la console)'}`);
  console.log(`  sites web : ${config.web.domains.join(', ')}\n`);
  const docs = await get<{ n: number }>("SELECT COUNT(*) n FROM documents WHERE source_type = 'doc' AND chunks > 0");
  if (config.autoIngest && Number(docs?.n ?? 0) === 0) {
    console.log('[ingest] base vide : indexation automatique des documents en arrière-plan…');
    void ingestAll();
  }
});
server.on('error', (e: NodeJS.ErrnoException) => {
  console.error(e.code === 'EADDRINUSE' ? `Le port ${config.port} est déjà utilisé (un autre serveur tourne ?).` : e);
  process.exit(1);
});
