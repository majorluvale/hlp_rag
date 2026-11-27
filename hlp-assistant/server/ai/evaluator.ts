import { config } from '../config';
import { now, run, uid } from '../db';
import { llm } from './llm';
import { getActivePrompt, render } from './prompts';
import { langfuseScore, type TraceSource } from './tracing';

/** LLM-as-a-judge: a first automatic grade, so reviewers can prioritise weak answers. */
export async function autoEvaluate(traceId: string, question: string, answer: string, sources: TraceSource[]) {
  if (!config.autoEval || !answer) return;
  try {
    const prompt = render(getActivePrompt('judge').content, {
      question,
      answer,
      sources: sources.map((s) => `[${s.n}] ${s.title}: ${s.snippet.slice(0, 500)}`).join('\n') || '(aucune)',
    });
    const res = await llm.chat.completions.create({
      model: config.llm.fastModel,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0,
      response_format: { type: 'json_object' },
    });
    const raw = res.choices[0]?.message?.content ?? '{}';
    const j = JSON.parse(raw.slice(raw.indexOf('{'), raw.lastIndexOf('}') + 1)) as Record<string, number | string>;
    const keys = ['groundedness', 'relevance', 'completeness', 'safety'];
    const vals = keys.map((k) => Number(j[k])).filter((v) => v >= 1 && v <= 5);
    if (!vals.length) return;
    const score = vals.reduce((a, b) => a + b, 0) / vals.length;
    run(
      "INSERT INTO feedback (id, trace_id, user_id, kind, rating, labels_json, comment, created_at) VALUES (?, ?, NULL, 'auto', ?, ?, ?, ?)",
      uid('fb_'), traceId, Math.round(score * 10) / 10, JSON.stringify(Object.fromEntries(keys.map((k) => [k, j[k]]))),
      String(j.comment ?? ''), now(),
    );
    langfuseScore(traceId, 'llm_judge', score, String(j.comment ?? ''));
  } catch (e) {
    console.warn('[eval]', (e as Error).message);
  }
}
