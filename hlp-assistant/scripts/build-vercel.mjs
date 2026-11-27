// Builds the Vercel deployment with the Build Output API (v3):
//   .vercel/output/static              → the React interface (CDN)
//   .vercel/output/functions/api.func  → the API bundled into one Node function (streaming enabled), with
//                                         the e5 embedding model and its ONNX runtime (Linux x64) inside,
//                                         so question embeddings are computed locally — no external API, no cost.
// Run by Vercel through `npm run vercel-build` (see vercel.json). Vercel builds on Linux, so the native
// binaries found in node_modules are the right ones.
import { build } from 'esbuild';
import { nodeFileTrace } from '@vercel/nft';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const out = path.resolve('.vercel/output');
fs.rmSync(out, { recursive: true, force: true });

// 1) static interface (built beforehand by `vite build` into dist/)
if (!fs.existsSync('dist/index.html')) throw new Error('dist/ manquant : lancez "vite build" avant ce script');
fs.cpSync('dist', path.join(out, 'static'), { recursive: true });

// 2) API function: our code + transformers.js bundled; native packages kept external and copied below
const fn = path.join(out, 'functions/api.func');
fs.mkdirSync(fn, { recursive: true });
const NATIVE = ['onnxruntime-node', 'onnxruntime-common', 'onnxruntime-web', 'sharp'];
await build({
  entryPoints: ['server/vercel.ts'],
  outfile: path.join(fn, 'index.mjs'),
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'esm',
  external: NATIVE,
  minify: false,
  legalComments: 'none',
  logLevel: 'warning',
  banner: {
    js: "import { createRequire as __cr } from 'node:module'; import { fileURLToPath as __fu } from 'node:url'; import { dirname as __dn } from 'node:path'; const require = __cr(import.meta.url); const __filename = __fu(import.meta.url); const __dirname = __dn(__filename);",
  },
});

// 3) native dependencies (+ everything they need), traced like Vercel does
const stub = path.join(root, '.vercel', 'nft-entry.mjs');
fs.writeFileSync(stub, "import 'onnxruntime-node';\nimport 'sharp';\n");
const { fileList } = await nodeFileTrace([stub], { base: root });
const wantedOrt = `${path.sep}linux${path.sep}x64${path.sep}`;
let copied = 0;
for (const rel of fileList) {
  if (!rel.startsWith('node_modules')) continue;
  const norm = path.normalize(rel);
  // keep only the Linux x64 ONNX runtime binaries (the package ships every platform: ~290 MB)
  if (norm.includes(`onnxruntime-node${path.sep}bin${path.sep}`) && !norm.includes(wantedOrt)) continue;
  const dest = path.join(fn, norm);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(path.join(root, norm), dest, { dereference: true });
  copied++;
}
// the ONNX binding loads shared libraries next to it at runtime (not visible to the tracer)
for (const dir of fs.globSync('node_modules/onnxruntime-node/bin/napi-v*/linux/x64')) {
  fs.cpSync(dir, path.join(fn, dir), { recursive: true });
}
fs.rmSync(stub);

// 4) the embedding model (same files as the local index was built with)
const MODEL = 'Xenova/multilingual-e5-small';
const FILES = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx'];
for (const f of FILES) {
  const dest = path.join(fn, 'models', MODEL, f);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const local = path.join('data/models', MODEL, f);
  if (fs.existsSync(local)) {
    fs.copyFileSync(local, dest);
    continue;
  }
  const res = await fetch(`https://huggingface.co/${MODEL}/resolve/main/${f}`);
  if (!res.ok) throw new Error(`Téléchargement du modèle impossible (${f}) : HTTP ${res.status}`);
  fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
}

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
      memory: 2048,
    },
    null,
    2,
  ),
);
fs.writeFileSync(path.join(fn, 'package.json'), JSON.stringify({ type: 'module' }));

// 5) routing: static files first, then /api/* → function, everything else → the single-page app
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

const dirSize = (d) =>
  fs.readdirSync(d, { withFileTypes: true }).reduce((s, e) => {
    const p = path.join(d, e.name);
    return s + (e.isDirectory() ? dirSize(p) : fs.statSync(p).size);
  }, 0);
const mb = (dirSize(fn) / 1024 / 1024).toFixed(0);
console.log(`[vercel] build prêt : .vercel/output — fonction API ${mb} Mo (${copied} fichiers natifs, modèle ${MODEL})`);
if (Number(mb) > 245) throw new Error(`Fonction trop volumineuse pour Vercel (${mb} Mo > 250 Mo)`);
