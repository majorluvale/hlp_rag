# HLP Assistant — RAG agentique Housing, Land & Property (TypeScript)

Assistant IA spécialisé **HLP / LTP** pour le NRC et ses partenaires. Il répond en citant ses sources, va chercher
l'information à jour sur les sites officiels, apprend des retours des utilisateurs et des corrections de l'équipe HLP,
et chaque réponse est traçable, évaluable et corrigeable dans un espace d'observabilité.

## Ce qui change par rapport à la version Python

| Avant (Python / Streamlit) | Maintenant (TypeScript) |
|---|---|
| Recherche vectorielle seule (Chroma, k=3) | **RAG hybride** : vecteurs multilingues + BM25 fusionnés (RRF), diversité par document |
| Une seule recherche | **Agent** : le modèle décide de relancer la recherche, de chercher en ligne, de lire une page |
| Aucune info récente | **Recherche web limitée aux sites de confiance** : AoR HLP (Global Protection Cluster), NRC, ONU-Habitat, Global CCCM Cluster, Global Shelter Cluster, ReliefWeb. Le nouveau cluster SLSC (Shelter, Land and Site Coordination) n’a pas encore de site propre : ajoutez son domaine à `WEB_DOMAINS` dès qu’il existe |
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

Par défaut, les embeddings tournent **en local** (modèle `Xenova/multilingual-e5-small`, ~120 Mo, téléchargé automatiquement
depuis Hugging Face au premier lancement) : aucun coût et aucun document envoyé à un tiers. Sans clé LLM dans le `.env`,
connectez-vous en admin et renseignez l'API dans *Observabilité › Paramètres*.

## Accès public sans ngrok : Cloudflare Tunnel (gratuit, sans compte)

```powershell
winget install --id Cloudflare.cloudflared
npm start
npm run tunnel            # affiche une URL https://xxxx.trycloudflare.com à partager
```

L'URL change à chaque lancement. Pour une URL fixe (ex. `hlp.mondomaine.org`), créez un tunnel nommé avec un compte
Cloudflare gratuit. Autres options : `npx localtunnel --port 3000`, ou déploiement du `Dockerfile` (Render, Railway, Fly.io,
Azure Container Apps) avec un volume persistant monté sur `/app/data`.

## Configurer l'API du LLM depuis l'application

Un administrateur peut changer de fournisseur LLM **sans redéployer** : *Observabilité › Paramètres*.

- Fournisseurs prêts à l'emploi : Groq, OpenAI, Anthropic (Claude), Mistral, OpenRouter, Azure OpenAI, Ollama, ou toute API compatible OpenAI.
- **Tester la connexion** vérifie la clé et le modèle, puis liste les modèles disponibles.
- La clé est **chiffrée** dans la base (AES-256-GCM, dérivée de `SESSION_SECRET`) et n'est jamais renvoyée au navigateur : seuls ses 4 derniers caractères sont affichés.
- Les valeurs du `.env` servent de configuration par défaut. « Revenir au .env » supprime la configuration de la console.
- ⚠ Si vous changez `SESSION_SECRET`, ressaisissez la clé dans la console.

## Héberger sur Vercel (en attendant un serveur)

Vercel est « serverless » : pas de disque persistant ni de processus permanent. L'application s'adapte :

| Élément | Local / Docker | Vercel |
|---|---|---|
| Base de données | fichier SQLite `data/hlp.db` | **Turso** (SQLite hébergé, gratuit) via `DATABASE_URL` |
| Embeddings des questions | modèle local (CPU) | **API Hugging Face**, même modèle `multilingual-e5-small` : l'index construit sur votre PC reste valable |
| Indexation des PDF | console ou `npm run ingest` | `npm run ingest` **depuis votre PC**, avec `DATABASE_URL` pointant vers Turso |
| Observabilité | identique : traces, revues, corrections et prompts sont dans la base partagée | identique ; le juge automatique et Langfuse terminent après la réponse (`waitUntil`) |

**Étapes**

1. Créez une base Turso : `turso db create hlp-assistant`, puis `turso db show hlp-assistant --url` et `turso db tokens create hlp-assistant`. Le site turso.tech permet aussi de tout faire sans CLI.
2. Indexez les documents dans Turso depuis votre PC. Dans PowerShell :
   ```powershell
   $env:DATABASE_URL="libsql://…turso.io"; $env:DATABASE_AUTH_TOKEN="…"; npm run ingest
   ```
3. Sur vercel.com, importez le dépôt GitHub. Choisissez **Root Directory = `hlp-assistant`** ; `vercel.json` fournit la commande de build.
4. Variables d'environnement Vercel : `DATABASE_URL`, `DATABASE_AUTH_TOKEN`, `SESSION_SECRET`, `ADMIN_EMAILS`, `EMBEDDING_PROVIDER=huggingface`, `HF_TOKEN`, `SMTP_*` (indispensable en ligne pour recevoir les codes), et optionnellement `TAVILY_API_KEY` et `LANGFUSE_*`.
5. Déployez, connectez-vous avec un email de `ADMIN_EMAILS`, puis renseignez l'API du LLM dans *Observabilité › Paramètres*.

Le build (`npm run vercel-build`) produit `.vercel/output` (Build Output API) :
- l'interface sur le CDN ;
- l'API dans une seule fonction Node, avec réponse en streaming et une durée maximale de 300 s.

Limites à connaître :
- démarrage à froid de quelques secondes (chargement de l'index) ;
- plan Hobby réservé à un usage non commercial : un compte Pro ou un serveur (Docker) convient mieux en production.

## Configuration principale (`.env`)

| Variable | Rôle |
|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Valeurs par défaut du LLM (Groq + `openai/gpt-oss-120b`), remplaçables dans *Observabilité › Paramètres* |
| `DATABASE_URL`, `DATABASE_AUTH_TOKEN` | Base Turso (obligatoire sur Vercel). Vide = SQLite local |
| `EMBEDDING_PROVIDER`, `HF_TOKEN` | `local` (défaut), `huggingface` (Vercel) ou `openai` |
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
SQLite local (node:sqlite) ou Turso (Vercel) : utilisateurs, sessions, OTP, conversations, documents, extraits, traces, retours, prompts, paramètres
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
