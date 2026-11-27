// Builds the Vercel deployment with the Build Output API (v3):
//   .vercel/output/static            → the React interface (CDN)
//   .vercel/output/functions/api.func → the whole API bundled into one Node function (streaming enabled)
// Run by Vercel through `npm run vercel-build` (see vercel.json).
import { build } from 'esbuild';
import fs from 'node:fs';
import path from 'node:path';

const out = path.resolve('.vercel/output');
fs.rmSync(out, { recursive: true, force: true });

// 1) static interface (built beforehand by `vite build` into dist/)
if (!fs.existsSync('dist/index.html')) throw new Error('dist/ manquant : lancez "vite build" avant ce script');
fs.cpSync('dist', path.join(out, 'static'), { recursive: true });

// 2) API function — everything bundled; local-only modules (node:sqlite, transformers.js, native libsql) stay lazy & external
const fn = path.join(out, 'functions/api.func');
fs.mkdirSync(fn, { recursive: true });
await build({
  entryPoints: ['server/vercel.ts'],
  outfile: path.join(fn, 'index.mjs'),
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'esm',
  minify: false,
  sourcemap: false,
  legalComments: 'none',
  logLevel: 'warning',
  banner: {
    js: "import { createRequire as __cr } from 'node:module'; import { fileURLToPath as __fu } from 'node:url'; import { dirname as __dn } from 'node:path'; const require = __cr(import.meta.url); const __filename = __fu(import.meta.url); const __dirname = __dn(__filename);",
  },
});
fs.writeFileSync(
  path.join(fn, '.vc-config.json'),
  JSON.stringify(
    {
      runtime: 'nodejs22.x',
      handler: 'index.mjs',
      launcherType: 'Nodejs',
      shouldAddHelpers: false,
      supportsResponseStreaming: true,
      maxDuration: 300,
    },
    null,
    2,
  ),
);
fs.writeFileSync(path.join(fn, 'package.json'), JSON.stringify({ type: 'module' }));

// 3) routing: static files first, then /api/* → function, everything else → the single-page app
fs.writeFileSync(
  path.join(out, 'config.json'),
  JSON.stringify(
    {
      version: 3,
      routes: [
        { handle: 'filesystem' },
        { src: '^/api/(.*)$', dest: '/api?__path=$1' },
        { src: '^/(.*)$', dest: '/index.html' },
      ],
    },
    null,
    2,
  ),
);

const size = (fs.statSync(path.join(fn, 'index.mjs')).size / 1024 / 1024).toFixed(1);
console.log(`[vercel] build prêt : .vercel/output (fonction API ${size} Mo)`);
