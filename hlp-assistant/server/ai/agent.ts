import type OpenAI from 'openai';
import { config } from '../config';
import { background } from '../background';
import { all, batch, parseJson, q, type UserProfile, type UserRow } from '../db';
import { findCorrections, retrieve, type Hit } from '../rag/retrieval';
import { fetchPage, learnFromPage, searchWeb } from '../rag/web';
import { autoEvaluate } from './evaluator';
import { getLlm } from './llm';
import { getActivePrompt, render } from './prompts';
import { Tracer, type TraceSource } from './tracing';

type Msg = OpenAI.Chat.Completions.ChatCompletionMessageParam;

export type AgentEvent =
  | { type: 'status'; text: string }
  | { type: 'token'; text: string }
  | { type: 'reset' }
  | { type: 'sources'; sources: TraceSource[] };

export interface AgentInput {
  user: UserRow | null;
  conversationId: string | null;
  question: string;
  history: { role: 'user' | 'assistant'; content: string }[];
  emit?: (e: AgentEvent) => void;
  promptOverride?: string;
  playground?: boolean;
  signal?: AbortSignal;
}

const TOOLS: OpenAI.Chat.Completions.ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'search_knowledge',
      description:
        "Recherche dans la base documentaire HLP interne (guidances NRC, AoR HLP, Protection, CCCM, Shelter, SLSC…). Utilise des mots-clés précis ; reformule en anglais si besoin.",
      parameters: {
        type: 'object',
        properties: { query: { type: 'string', description: 'Requête de recherche' } },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_web',
      description: `Recherche des informations à jour sur les sites officiels de confiance uniquement : ${config.web.domains.join(', ')}. À utiliser pour les actualités, la réforme des clusters, les contextes pays, les publications récentes.`,
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Requête (de préférence en anglais)' },
          site: { type: 'string', description: `Optionnel : restreindre à un domaine parmi ${config.web.domains.join(', ')}` },
        },
        required: ['query'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_webpage',
      description: 'Lit le contenu d’une page web ou d’un PDF trouvé via search_web (domaines de confiance uniquement).',
      parameters: {
        type: 'object',
        properties: { url: { type: 'string' } },
        required: ['url'],
      },
    },
  },
];

/** Numbers every source shown to the model so citations [n] can be resolved and graded. */
class SourceRegistry {
  private items: TraceSource[] = [];
  private byKey = new Map<string, number>();

  add(s: Omit<TraceSource, 'n'>, key: string): number {
    const existing = this.byKey.get(key);
    if (existing) return existing;
    const n = this.items.length + 1;
    this.items.push({ ...s, n });
    this.byKey.set(key, n);
    return n;
  }
  addHit(h: Hit) {
    return this.add(
      { kind: 'doc', docId: h.docId, chunkId: h.chunkId, title: h.title, url: h.url, page: h.page, score: h.score, snippet: h.text },
      `c${h.chunkId}`,
    );
  }
  finalize(answer: string): TraceSource[] {
    const cited = new Set([...answer.matchAll(/\[(\d{1,3})\]/g)].map((m) => Number(m[1])));
    return this.items.map((s) => ({ ...s, cited: cited.has(s.n) }));
  }
}

function formatHits(hits: Hit[], reg: SourceRegistry): string {
  if (!hits.length) return '(aucun extrait pertinent trouvé)';
  return hits
    .map((h) => {
      const n = reg.addHit(h);
      const where = h.page ? `, p. ${h.page}` : '';
      return `[${n}] ${h.title}${where}${h.sourceType === 'web' && h.url ? ` (${h.url})` : ''}\n${h.text}`;
    })
    .join('\n\n');
}

export async function buildUserContext(user: UserRow | null): Promise<string> {
  if (!user) return '- Utilisateur de test (playground)';
  const p = parseJson<UserProfile>(user.profile_json, {});
  const lines = [
    `- Statut : ${user.is_guest ? 'invité (non connecté)' : 'connecté'}${user.role === 'reviewer' || user.role === 'admin' ? ' — membre de l’équipe HLP' : ''}`,
    user.display_name && `- Nom / pseudo : ${user.display_name}`,
    p.jobTitle && `- Fonction : ${p.jobTitle}`,
    p.organisation && `- Organisation : ${p.organisation}`,
    p.sector && `- Secteur : ${p.sector}`,
    p.country && `- Pays / opération : ${p.country}`,
    p.expertise && `- Niveau en HLP : ${p.expertise}`,
    p.language && `- Langue préférée : ${p.language}`,
    p.interests && `- Centres d'intérêt : ${p.interests}`,
  ].filter(Boolean) as string[];
  const recent = await all<{ title: string }>(
    'SELECT title FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 5',
    user.id,
  );
  if (recent.length) lines.push(`- Sujets récents : ${recent.map((r) => r.title).join(' ; ')}`);
  if (lines.length === 1) lines.push('- Profil non renseigné : reste accessible, demande le contexte (pays, rôle) si cela change la réponse.');
  return lines.join('\n');
}

export async function runAgent(input: AgentInput) {
  const emit = input.emit ?? (() => {});
  const [{ client, cfg }, system, userContext] = await Promise.all([getLlm(), getActivePrompt('system'), buildUserContext(input.user)]);
  if (!cfg.apiKey && !/localhost|127\.0\.0\.1/.test(cfg.baseURL)) {
    throw Object.assign(new Error('Aucune clé API LLM configurée. Un administrateur doit la renseigner dans Observabilité → Paramètres.'), { userFacing: true });
  }
  const tracer = await Tracer.start({
    userId: input.user?.id ?? null,
    conversationId: input.conversationId,
    question: input.question,
    userContext,
    promptVersionId: input.promptOverride ? null : system.id,
    model: cfg.model,
    playground: input.playground,
  });
  const reg = new SourceRegistry();
  let answer = '';
  const maxSteps = Math.min(10, Math.max(1, cfg.maxSteps || 6));

  try {
    // 1) Retrieval (hybrid + learned rewards) and reviewer corrections memory
    emit({ type: 'status', text: 'Recherche dans la base de connaissances HLP…' });
    const sp = tracer.span('retrieval', 'initial_retrieval', { query: input.question });
    const { hits, qvec } = await retrieve(input.question, 6);
    const corrections = await findCorrections(qvec);
    sp.end(
      hits.map((h) => ({ title: h.title, page: h.page, score: +h.score.toFixed(5), cosine: +h.cosine.toFixed(3), reward: +h.reward.toFixed(2) })),
      { corrections: corrections.map((c) => ({ id: c.id, similarity: +c.similarity.toFixed(3) })) },
    );
    if (corrections.length) {
      tracer.usedCorrections = true;
      await batch(corrections.map((c) => q('UPDATE corrections SET hits = hits + 1 WHERE id = ?', c.id)));
    }
    const correctionsText = corrections.length
      ? corrections
          .map((c) => {
            const n = reg.add({ kind: 'correction', title: 'Réponse validée par l’équipe HLP', snippet: c.correctedAnswer }, `co${c.id}`);
            return `[${n}] Question similaire : « ${c.question} »\nRéponse validée : ${c.correctedAnswer}`;
          })
          .join('\n\n')
      : '(aucune)';
    const knowledge = formatHits(hits, reg);

    const systemText = render(input.promptOverride ?? system.content, {
      date: new Date().toLocaleDateString('fr-FR', { dateStyle: 'full' }),
      user_context: userContext,
      corrections: correctionsText,
      knowledge,
      web_domains: config.web.domains.join(', '),
    });

    const messages: Msg[] = [
      { role: 'system', content: systemText },
      ...input.history.slice(-10).map((m) => ({ role: m.role, content: m.content.slice(0, 4000) }) as Msg),
      { role: 'user', content: input.question },
    ];

    // 2) Agent loop: the model decides to answer or to use tools (more retrieval, web search, page reading)
    let toolsEnabled = true;
    for (let step = 0; step < maxSteps; step++) {
      const lastStep = step === maxSteps - 1;
      const useTools = toolsEnabled && !lastStep;
      emit({ type: 'status', text: step === 0 ? 'Rédaction de la réponse…' : 'Analyse des nouvelles sources…' });
      const gen = tracer.span('llm', `generation_${step + 1}`, { messages: step === 0 ? messages : messages.slice(-4) });
      let content = '';
      const calls: { id: string; name: string; args: string }[] = [];
      type Usage = { prompt_tokens?: number; completion_tokens?: number };
      let usage: Usage | null = null;
      try {
        const stream = await client.chat.completions.create(
          {
            model: cfg.model,
            messages,
            temperature: cfg.temperature,
            max_tokens: 2500,
            stream: true,
            stream_options: { include_usage: true },
            ...(useTools ? { tools: TOOLS, tool_choice: 'auto' as const } : {}),
          },
          { signal: input.signal },
        );
        for await (const chunk of stream) {
          const delta = chunk.choices?.[0]?.delta;
          if (delta?.content) {
            content += delta.content;
            emit({ type: 'token', text: delta.content });
          }
          for (const tc of delta?.tool_calls ?? []) {
            const c = (calls[tc.index] ??= { id: '', name: '', args: '' });
            if (tc.id) c.id = tc.id;
            if (tc.function?.name) c.name += tc.function.name;
            if (tc.function?.arguments) c.args += tc.function.arguments;
          }
          const u: Usage | null | undefined = chunk.usage ?? (chunk as unknown as { x_groq?: { usage?: Usage } }).x_groq?.usage;
          if (u) usage = u;
        }
      } catch (e) {
        gen.fail(e);
        if (input.signal?.aborted) throw e;
        if (useTools && /tool|function/i.test((e as Error).message)) {
          // malformed tool call from the model: retry this step without tools
          toolsEnabled = false;
          if (content) emit({ type: 'reset' });
          continue;
        }
        throw e;
      }
      tracer.addUsage(usage);
      const validCalls = calls.filter(Boolean).filter((c) => c.name);
      gen.end({ content, tool_calls: validCalls.map((c) => ({ name: c.name, args: c.args })) }, { usage, model: cfg.model });

      if (!validCalls.length) {
        answer = content;
        break;
      }
      if (content) emit({ type: 'reset' });
      messages.push({
        role: 'assistant',
        content: content || null,
        tool_calls: validCalls.map((c, i) => ({
          id: c.id || `call_${step}_${i}`,
          type: 'function' as const,
          function: { name: c.name, arguments: c.args || '{}' },
        })),
      });
      for (const [i, c] of validCalls.entries()) {
        const result = await executeTool(c.name, parseJson<Record<string, string>>(c.args, {}), reg, tracer, emit);
        messages.push({ role: 'tool', tool_call_id: c.id || `call_${step}_${i}`, content: result });
      }
    }

    if (!answer.trim()) answer = "Désolé, je n'ai pas pu formuler de réponse. Pouvez-vous reformuler votre question ?";
    const sources = reg.finalize(answer);
    emit({ type: 'sources', sources: sources.filter((s) => s.cited) });
    await tracer.finish({ answer, sources });
    background(autoEvaluate(tracer.id, input.question, answer, sources.filter((s) => s.cited)));
    return { answer, traceId: tracer.id, sources: sources.filter((s) => s.cited) };
  } catch (e) {
    const msg = (e as Error).message;
    await tracer.finish({ answer, sources: reg.finalize(answer), error: msg }).catch(() => {});
    throw Object.assign(new Error(msg), { traceId: tracer.id });
  }
}

async function executeTool(
  name: string,
  args: Record<string, string>,
  reg: SourceRegistry,
  tracer: Tracer,
  emit: (e: AgentEvent) => void,
): Promise<string> {
  const sp = tracer.span('tool', name, args);
  try {
    if (name === 'search_knowledge') {
      emit({ type: 'status', text: `Recherche approfondie : « ${args.query} »` });
      const { hits } = await retrieve(String(args.query ?? ''), 5);
      sp.end(hits.map((h) => ({ title: h.title, page: h.page, score: +h.score.toFixed(5) })));
      return formatHits(hits, reg);
    }
    if (name === 'search_web') {
      tracer.usedWeb = true;
      emit({ type: 'status', text: `Recherche en ligne${args.site ? ` sur ${args.site}` : ''} : « ${args.query} »` });
      const { provider, results } = await searchWeb(String(args.query ?? ''), args.site);
      sp.end(results, { provider });
      if (!results.length) return 'Aucun résultat sur les sites de confiance.';
      return results
        .map((r) => {
          const n = reg.add({ kind: 'web', title: r.title, url: r.url, snippet: r.snippet }, `w${r.url}`);
          return `[${n}] ${r.title}\nURL : ${r.url}\n${r.snippet}`;
        })
        .join('\n\n');
    }
    if (name === 'read_webpage') {
      tracer.usedWeb = true;
      const url = String(args.url ?? '');
      emit({ type: 'status', text: `Lecture de ${new URL(url).hostname}…` });
      const page = await fetchPage(url);
      learnFromPage(page);
      const n = reg.add({ kind: 'web', title: page.title, url: page.url, snippet: page.text.slice(0, 1200) }, `w${page.url}`);
      sp.end({ title: page.title, chars: page.text.length, cached: page.cached });
      return `[${n}] ${page.title} (${page.url})\n${page.text.slice(0, 12000)}`;
    }
    sp.fail(`Outil inconnu : ${name}`);
    return `Outil inconnu : ${name}`;
  } catch (e) {
    sp.fail(e);
    return `Erreur de l'outil ${name} : ${(e as Error).message}`;
  }
}
