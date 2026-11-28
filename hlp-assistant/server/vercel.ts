import type { IncomingMessage, ServerResponse } from 'node:http';
import { app, ready } from './app';

/**
 * Vercel serverless entry point (bundled by scripts/build-vercel.mjs).
 * Every /api/* request is routed here; the original path is passed in ?__path= and restored.
 */
export default async function handler(req: IncomingMessage, res: ServerResponse) {
  try {
    await ready;
  } catch (e) {
    res.statusCode = 500;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({ error: `Initialisation impossible : ${(e as Error).message}` }));
    return;
  }
  const url = new URL(req.url ?? '/', 'http://localhost');
  const p = url.searchParams.get('__path');
  if (p !== null) {
    url.searchParams.delete('__path');
    const qs = url.searchParams.toString();
    req.url = `/api/${p}${qs ? `?${qs}` : ''}`;
  }
  return (app as unknown as (req: IncomingMessage, res: ServerResponse) => void)(req, res);
}
