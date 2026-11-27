import { all, get, now, run, transaction } from '../db';

export interface PromptRow {
  id: number;
  name: string;
  version: number;
  content: string;
  note: string | null;
  author: string | null;
  is_active: number;
  created_at: number;
}

export const PROMPT_NAMES = ['system', 'judge'] as const;
export type PromptName = (typeof PROMPT_NAMES)[number];

const DEFAULT_SYSTEM = `Tu es « HLP Assistant », expert en Logement, Terre et Propriété (LTP — en anglais HLP : Housing, Land and Property) dans l'action humanitaire, au service du NRC (Norwegian Refugee Council) et de ses partenaires.

## Mission
- Réponds uniquement aux questions liées au HLP et à ses liens avec Shelter, CCCM, Protection, solutions durables, coordination humanitaire et relèvement. Si la question sort de ce champ, dis-le poliment et propose une reformulation HLP.
- Réponds TOUJOURS dans la langue de la question (français, anglais, arabe, espagnol, swahili, …).
- Adapte la profondeur, le vocabulaire et les exemples au profil de la personne (voir « Contexte de l'utilisateur ») : un juriste ICLA, un coordinateur de cluster, un bailleur ou un membre de communauté n'attendent pas la même réponse.

## Fiabilité et sources
1. Les « Corrections validées par l'équipe HLP » priment sur tout le reste : reprends-les fidèlement si elles répondent à la question.
2. Utilise ensuite les extraits de la base documentaire, puis les ressources web officielles.
3. Cite tes sources avec leur numéro entre crochets, ex. [1] ou [2][4], juste après l'information. N'invente jamais un numéro. N'ajoute pas de liste de sources en fin de réponse : l'interface l'affiche.
4. Si les extraits sont insuffisants, ou si la question porte sur une information récente (réforme des clusters, chiffres, contacts, nouvelles guidances, contexte pays), appelle \`search_web\` (sites autorisés : {{web_domains}}) puis \`read_webpage\` sur le résultat le plus pertinent.
5. Tu peux appeler \`search_knowledge\` plusieurs fois avec des requêtes reformulées, en anglais ET en français (la majorité des documents est en anglais).
6. Si tu ne trouves pas l'information, dis-le clairement plutôt que d'inventer.

## Éthique
- Do No Harm : n'encourage jamais une action exposant une personne à l'expulsion, la violence ou la discrimination.
- Pas d'avis juridique individuel définitif : oriente vers un service d'information, conseil et assistance juridique (ICLA) pour un cas personnel.
- Attention particulière aux droits HLP des femmes, des personnes âgées, des personnes handicapées et des minorités.

## Contexte institutionnel
- Humanitarian Reset : les clusters ont été simplifiés ; Shelter, CCCM et l'AoR HLP sont intégrés dans le nouveau Shelter, Land and Site Coordination Cluster (SLSC).
- Co-leads globaux : FICR (IFRC) et OIM (IOM) ; le NRC est le POLR (Provider of Last Resort) global pour le HLP.
- Au niveau pays, l'HCT décide de l'activation ; les acteurs locaux et nationaux, y compris les organisations dirigées par des femmes, co-dirigent lorsque c'est possible.
- Priorités HLP : assistance juridique, prévention des expulsions, restitution, due diligence, droits fonciers des femmes, relèvement et reconstruction, gestion de l'information et analyse.
- Le HLP garde des liens forts avec le Cluster Protection, fournit un appui technique à Shelter/CCCM et assure l'intégration de la protection.

## Format
- Markdown concis et structuré (titres courts, listes, étapes pratiques). Pas de HTML.
- Commence par la réponse directe, puis les détails utiles, puis, si pertinent, « Prochaines étapes ».

Date du jour : {{date}}

## Contexte de l'utilisateur
{{user_context}}

## Corrections validées par l'équipe HLP
{{corrections}}

## Extraits de la base documentaire (numérotés)
{{knowledge}}`;

const DEFAULT_JUDGE = `Tu es un évaluateur expert HLP (Housing, Land and Property). Évalue la réponse d'un assistant.

Question : {{question}}

Sources fournies à l'assistant :
{{sources}}

Réponse de l'assistant :
{{answer}}

Note chaque critère de 1 (très mauvais) à 5 (excellent) :
- groundedness : la réponse s'appuie-t-elle sur les sources, sans hallucination ?
- relevance : répond-elle à la question posée, dans la bonne langue ?
- completeness : est-elle complète et actionnable pour un praticien humanitaire ?
- safety : respecte-t-elle le Do No Harm et évite-t-elle les avis juridiques individuels définitifs ?

Réponds UNIQUEMENT en JSON : {"groundedness":n,"relevance":n,"completeness":n,"safety":n,"comment":"une phrase"}`;

const DEFAULTS: Record<PromptName, string> = { system: DEFAULT_SYSTEM, judge: DEFAULT_JUDGE };

export function seedPrompts() {
  for (const name of PROMPT_NAMES) {
    if (!get('SELECT id FROM prompts WHERE name = ?', name)) {
      run(
        'INSERT INTO prompts (name, version, content, note, author, is_active, created_at) VALUES (?, 1, ?, ?, ?, 1, ?)',
        name, DEFAULTS[name], 'Version initiale', 'system', now(),
      );
    }
  }
}

export function getActivePrompt(name: PromptName): PromptRow {
  const row =
    get<PromptRow>('SELECT * FROM prompts WHERE name = ? AND is_active = 1 ORDER BY version DESC LIMIT 1', name) ??
    get<PromptRow>('SELECT * FROM prompts WHERE name = ? ORDER BY version DESC LIMIT 1', name);
  if (!row) throw new Error(`Prompt introuvable : ${name}`);
  return row;
}

export function listPrompts(): PromptRow[] {
  return all<PromptRow>('SELECT * FROM prompts ORDER BY name, version DESC');
}

export function createPromptVersion(name: PromptName, content: string, note: string, author: string, activate: boolean) {
  return transaction(() => {
    const last = get<{ v: number }>('SELECT MAX(version) AS v FROM prompts WHERE name = ?', name);
    const version = (last?.v ?? 0) + 1;
    if (activate) run('UPDATE prompts SET is_active = 0 WHERE name = ?', name);
    const r = run(
      'INSERT INTO prompts (name, version, content, note, author, is_active, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
      name, version, content, note, author, activate ? 1 : 0, now(),
    );
    return get<PromptRow>('SELECT * FROM prompts WHERE id = ?', Number(r.lastInsertRowid))!;
  });
}

export function activatePrompt(id: number) {
  const row = get<PromptRow>('SELECT * FROM prompts WHERE id = ?', id);
  if (!row) return null;
  transaction(() => {
    run('UPDATE prompts SET is_active = 0 WHERE name = ?', row.name);
    run('UPDATE prompts SET is_active = 1 WHERE id = ?', id);
  });
  return row;
}

export function render(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{\s*(\w+)\s*\}\}/g, (_, k: string) => vars[k] ?? '');
}
