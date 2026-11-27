# HLP Assistant — RAG agentique Housing, Land & Property (TypeScript)

Assistant IA spécialisé **HLP / LTP** pour le NRC et ses partenaires. Il répond en citant ses sources, va chercher
l'information à jour sur les sites officiels, apprend des retours des utilisateurs et des corrections de l'équipe HLP,
et chaque réponse est traçable, évaluable et corrigeable dans un espace d'observabilité.

## Ce qui change par rapport à la version Python

| Avant (Python / Streamlit) | Maintenant (TypeScript) |
|---|---|
| Recherche vectorielle seule (Chroma, k=3) | **RAG hybride** : vecteurs multilingues + BM25 fusionnés (RRF), diversité par document |
| Une seule recherche | **Agent** : le modèle décide de relancer la recherche, de chercher en ligne, de lire une page |
| Aucune info récente | **Recherche web limitée aux sites de confiance** : AoR HLP (Global Protection Cluster), NRC, ONU-Habitat, CCCM Cluster, Shelter Cluster / SLSC, ReliefWeb |
| Pas d'apprentissage | **Apprentissage par renforcement à partir des retours** (voir plus bas) + mémoire des corrections + pages web apprises |
| Langfuse en option | **Observabilité intégrée** : traces étape par étape, évaluation 1–5, corrections, LLM juge, tableau de bord, édition et test des prompts versionnés (+ miroir Langfuse optionnel) |
| Pas d'utilisateurs | **Invités sans connexion** ou connexion **email + code OTP** (sans mot de passe), nom ou pseudo, **contexte de l'utilisateur** (fonction, organisation, pays, niveau…) injecté dans chaque réponse |
| Thème Streamlit | Interface **style LinkedIn** (clair/sombre, mobile) |

## Démarrage

Prérequis : Node.js ≥ 22.13 (testé avec Node 26).

```powershell
cd hlp-assistant
npm install
copy .env.example .env    # puis renseignez LLM_API_KEY, SESSION_SECRET, ADMIN_EMAILS, SMTP_*
npm run ingest            # indexe les PDF (../../docs, ../../knowledge, data/docs)
npm run build             # construit l'interface
npm start                 # http://localhost:3000
```

En développement : `npm run dev` (API sur 3000 + interface Vite avec rechargement à chaud sur http://localhost:5173).

Les embeddings tournent **en local** (modèle `Xenova/multilingual-e5-small`, téléchargé automatiquement depuis Hugging Face
au premier lancement, ~120 Mo) : aucun coût et aucun document envoyé à un tiers pour l'indexation.

## Accès public sans ngrok : Cloudflare Tunnel (gratuit, sans compte)

```powershell
winget install --id Cloudflare.cloudflared
npm start
npm run tunnel            # affiche une URL https://xxxx.trycloudflare.com à partager
```

L'URL change à chaque lancement. Pour une URL fixe (ex. `hlp.mondomaine.org`), créez un tunnel nommé avec un compte
Cloudflare gratuit. Autres options : `npx localtunnel --port 3000`, ou déploiement du `Dockerfile` (Render, Railway, Fly.io,
Azure Container Apps) avec un volume persistant monté sur `/app/data`.

## Configuration principale (`.env`)

| Variable | Rôle |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Toute API compatible OpenAI. Par défaut Groq + `openai/gpt-oss-120b` (comme la version Python). Fonctionne aussi avec OpenAI, Azure OpenAI, Mistral, Ollama… |
| `LLM_FAST_MODEL` | Modèle de l'évaluateur automatique (LLM juge) |
| `ADMIN_EMAILS` | Emails qui deviennent administrateurs à la connexion |
| `SMTP_*` | Envoi des codes OTP (Gmail avec mot de passe d'application, Office 365, Brevo…). Sans SMTP, le code s'affiche dans la console du serveur |
| `DEV_SHOW_OTP` | `true` affiche aussi le code dans l'interface — **démo locale uniquement** |
| `WEB_DOMAINS` | Sites autorisés pour la recherche en ligne |
| `TAVILY_API_KEY` | Moteur de recherche recommandé (gratuit jusqu'à 1000 req./mois). Sinon DuckDuckGo |
| `DOCS_DIRS` | Dossiers de PDF à indexer |
| `LANGFUSE_*` | Optionnel : envoie aussi les traces et les scores vers Langfuse |

## Architecture

```
Navigateur (React, style LinkedIn)
   │  SSE (réponse en streaming + étapes : « Recherche sur nrc.no… »)
   ▼
Express (TypeScript)
   ├─ auth      invité automatique (cookie) · email + OTP · fusion de l'historique invité → compte
   ├─ agent     boucle d'outils : search_knowledge · search_web · read_webpage (max 6 étapes)
   │             └─ prompt système versionné + contexte utilisateur + corrections validées + extraits numérotés
   ├─ rag       index hybride en mémoire (cosinus + BM25 + RRF) × fiabilité apprise des sources
   ├─ learning  récompenses Beta(α, β) par document · mémoire des corrections · pages web apprises
   ├─ tracing   traces/étapes en SQLite (+ Langfuse) · LLM juge automatique
   └─ admin     tableau de bord · traces & revue · corrections · prompts · connaissances · utilisateurs
SQLite (node:sqlite, fichier data/hlp.db) : utilisateurs, sessions, OTP, conversations, documents, extraits, traces, retours, prompts
```

## Comment l'assistant apprend

1. **Renforcement des sources** — chaque document a une fiabilité Beta(α, β). Un 👍 utilisateur (+0,5) ou une note équipe
   ≥ 4 (+1,5) renforce les documents cités ; un 👎 ou une note ≤ 2 les pénalise ; une source cochée « non pertinente » par un
   relecteur est fortement pénalisée (+3 β). Le classement de la recherche est multiplié par cette fiabilité : les bonnes
   sources remontent, les mauvaises descendent, sans réentraîner de modèle.
2. **Mémoire des corrections** — quand un relecteur écrit la réponse attendue, elle est indexée. Pour toute question
   sémantiquement proche, elle est injectée en priorité (« Corrections validées par l'équipe HLP ») et affichée comme source
   « Validé par l'équipe ». Les corrections sont modifiables et désactivables.
3. **Apprentissage web** — les pages officielles lues par l'agent sont découpées, vectorisées et ajoutées à la base : la
   prochaine fois, l'information est trouvée localement.
4. **Prompts pilotés par l'équipe** — le tableau de bord compare satisfaction, note équipe et note automatique par version
   de prompt ; on teste une variante dans le playground avant de l'activer.

## Rôles

- **Invité** : pose des questions sans compte, peut renseigner son contexte, voter.
- **Utilisateur** (email + OTP) : retrouve ses conversations sur tous ses appareils.
- **Relecteur** (`reviewer`) : accède à l'onglet **Observabilité** (traces, revue, corrections, prompts).
- **Admin** : en plus, gère les rôles et la base documentaire. Les emails listés dans `ADMIN_EMAILS` sont admins.

## Sécurité

Codes OTP hachés (HMAC), 10 min de validité, 5 essais, délai entre deux envois ; sessions en cookie `httpOnly`
(`secure` derrière HTTPS) stockées hachées ; recherche et lecture web limitées à la liste blanche de domaines ;
limite de questions par heure pour les invités. Les réponses ne constituent pas un conseil juridique individuel.
