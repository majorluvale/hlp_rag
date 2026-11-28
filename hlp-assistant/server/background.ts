import { waitUntil } from '@vercel/functions';

/**
 * Runs work after the response (LLM judge, Langfuse export, learning web pages).
 * On Vercel, waitUntil keeps the function alive until it finishes; elsewhere it simply runs.
 */
export function background(p: Promise<unknown>) {
  const safe = p.catch((e) => console.warn('[background]', (e as Error)?.message ?? e));
  try {
    waitUntil(safe);
  } catch {
    /* not on Vercel */
  }
}
