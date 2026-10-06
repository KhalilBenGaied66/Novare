# Novare DossierOps

[![CI](https://github.com/KhalilBenGaied66/Novare/actions/workflows/ci.yml/badge.svg)](https://github.com/KhalilBenGaied66/Novare/actions/workflows/ci.yml)

*English summary [at the end of this page](#in-english).*

Assistant de traitement des demandes clients pour une entreprise de maintenance
(chauffage, ventilation, climatisation). Chaque demande est d'abord triée par des règles
métier, puis traitée par la voie la plus simple possible : une règle sans IA, une réponse
documentaire citée, un agent qui prépare un dossier, ou une transmission à une personne.

**Pile technique.** Python 3.11, FastAPI, LangGraph, LiteLLM (modèles hébergés, ou locaux
via Ollama), recherche BM25 et vecteurs (fastembed, Qdrant), SQLAlchemy (SQLite ou
PostgreSQL), Streamlit, Docker, GitHub Actions.

> **Contexte.** Novare Services est une entreprise fictive. Les documents, les clients
> et le jeu d'évaluation sont inventés pour ce prototype. Aucun utilisateur réel ne l'a
> testé.

## Ce que fait le système

| Demande | Voie | Ce qui se passe | IA générative |
|---|---|---|---|
| « Je conteste la facture de 120 € » (client connu) | `automation` | Ticket standard créé, doublons détectés | aucune |
| « Quelle est la majoration le week-end ? » | `rag` | Réponse tirée des documents, avec ses sources | facultative |
| « La chaudière est en panne » (client connu) | `agent` | Contrat vérifié, délai calculé, brouillon de réponse, ticket **proposé** | facultative |
| « Je veux résilier, mon avocat vous écrira » | `human` | Aucune réponse automatique, demande signalée pour un gestionnaire | aucune |

Trois principes structurent l'ensemble :

- **L'IA n'est pas la réponse par défaut.** Le tri est déterministe (aucun modèle) : la
  même demande prend toujours la même voie, et la règle appliquée est écrite dans le
  journal de décision.
- **Dans le doute, une personne décide.** Sujet sensible, sources insuffisantes, réponse
  sans citation, erreur interne : la demande est transmise. L'agent ne crée jamais de
  ticket ; il le propose, et un gestionnaire le valide en confirmant la priorité. Seule
  la règle des petits litiges crée un ticket sans validation.
- **Sans modèle, le système répond quand même.** Les réponses documentaires citent
  alors les passages tels quels et l'agent suit un plan fixe. Avec un modèle, hébergé ou
  local, celui-ci rédige sous les mêmes contrôles ; s'il échoue, la voie sans modèle
  reprend la main.

Le parcours d'une demande ([docs/03-architecture.md](docs/03-architecture.md)) :

```mermaid
flowchart TD
    A[POST /api/v1/ask] --> B[Masquage des données personnelles]
    B --> C{Tri par règles<br/>sans modèle}
    C -->|RG-04, RG-03b| H[Transmission à un gestionnaire]
    C -->|RG-03| AU[Ticket standard<br/>détection de doublon]
    C -->|DEFAULT| R[Recherche hybride]
    C -->|RG-05| AG[Agent à outils]
    R --> G{Confiance suffisante ?}
    G -->|non| H
    G -->|oui| GEN[Réponse citée<br/>LLM ou extraits]
    GEN -->|sans citation valide| H
    AG --> P[Brouillon + ticket proposé]
    P --> V{Validation humaine}
    V -->|approuvé| T[Ticket créé]
    AU --> L[(Journal des demandes)]
    GEN --> L
    P --> L
    H --> L
```

## État du projet

| | |
|---|---|
| **Fonctionne et testé** | Tri par règles, automatisation des petits litiges, recherche hybride (BM25 + vecteurs), réponses citées, agent à outils (LangGraph) avec validation humaine, masquage des données personnelles, contrats réservés au client du champ client (identité déclarée, non authentifiée), API (FastAPI), base de données, interface (Streamlit), évaluation sur 136 cas |
| **Exécuté avec de vrais modèles** | Rédaction des réponses, agent qui choisit ses outils, juge de fidélité : mesurés avec deux modèles ouverts exécutés localement (Qwen 3.5 via Ollama), voir les résultats ci-dessous. Les fournisseurs hébergés (Mistral, Anthropic) passent par le même code et n'ont pas été appelés, faute de clé : `scripts/smoke_llm.py` sert à le vérifier. |
| **Exécuté en CI seulement** | Images Docker et `docker-compose.yml` : à chaque push, la CI lance les tests et l'évaluation, puis démarre la pile complète (API, interface, PostgreSQL 17, Qdrant), pose une question par la recherche hybride et ouvre un ticket. Sur la machine de développement, tout tourne avec SQLite et un index de vecteurs local. |
| **Non fait** | Authentification des personnes et rôles, file de traitement des demandes transmises, migrations de schéma, purge des données, traces LLM, déploiement, test avec de vrais utilisateurs. Voir [docs/08-industrialisation.md](docs/08-industrialisation.md). |

## Démarrage

Python 3.11. Aucune clé ni conteneur n'est nécessaire.

Linux / macOS :

```bash
python3.11 -m venv .venv && . .venv/bin/activate
make install
make api      # http://localhost:8000/docs
make front    # http://localhost:8501 (dans un second terminal)
```

Windows (PowerShell) :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r backend/requirements-dev.txt
.\.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
# second terminal
.\.venv\Scripts\python -m streamlit run frontend/streamlit_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Au premier démarrage, l'API construit l'index des documents et télécharge le modèle
d'embeddings (environ 220 Mo, dans `data/models/`). Pour s'en passer : `RETRIEVAL_MODE=bm25`.

Une demande en ligne de commande :

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "Quelle est la majoration le week-end ?"}'
```

Pour utiliser un modèle, copier `.env.example` en `.env` et renseigner la clé du
fournisseur (`MISTRAL_API_KEY` pour les réponses documentaires, `ANTHROPIC_API_KEY` pour
l'agent, ou d'autres modèles via `RAG_MODEL` et `AGENT_MODEL`), puis vérifier avec
`scripts/smoke_llm.py` (commande ci-dessous). Sans clé : des modèles ouverts servis
localement par [Ollama](https://ollama.com), avec le bloc prêt à décommenter de
`.env.example`.

Avec Docker (vérifié en CI) : `POSTGRES_PASSWORD=... docker compose up --build`.

## Résultats mesurés

Évaluation du pipeline réel sur 136 cas (`evals/golden_set.json`). Les deux premières
colonnes sont sans LLM. Dans la troisième, les réponses sont rédigées par deux modèles
ouverts exécutés localement sur une carte graphique grand public (Qwen 3.5, à 4 et
9 milliards de paramètres, via Ollama). Rapports :
[hybrid.md](evals/reports/hybrid.md), [bm25.md](evals/reports/bm25.md) et
[hybrid-llm.md](evals/reports/hybrid-llm.md).

| Mesure | Recherche hybride | BM25 seul | Hybride + modèles locaux |
|---|---|---|---|
| Voie choisie par le tri | 98,5 % | 98,5 % | 98,5 % |
| Voie finale (après garde-fous) | 90,4 % | 90,4 % | 91,9 % |
| Document attendu parmi les 4 sources | 100 % | 98,6 % | 100 % |
| Rang du document attendu (MRR, 1 = toujours premier) | 0,89 | 0,86 | 0,89 |
| Document attendu cité, sur les demandes à répondre | 89,9 % | 88,4 % | 100 % |
| Faits attendus présents dans la réponse | 87,1 % | 87,1 % | 87,1 % |
| Réponse jugée fidèle aux sources par un second modèle | — | — | 98,6 % (69 sur 70) |
| Document d'un autre client atteint | 0 | 0 | 0 |

Pour lire ces chiffres :

- **Le tri repose sur du vocabulaire.** Une partie des cas a été écrite à partir des
  mêmes règles que le code : ils mesurent la conformité, pas la tenue face à de vrais
  courriers. Huit cas ont été écrits après coup avec l'attendu métier, sans être essayés
  avant la mesure : quatre échouent (« Nous comptons arrêter notre contrat » n'est pas
  reconnu comme une résiliation).
- **Sans LLM, la réponse cite des passages par recoupement de mots.** Elle ne reformule
  pas : une paraphrase, ou une question hors sujet qui partage deux mots du domaine
  avec un passage, reçoit un extrait qui ne répond pas. 6 des 22 questions hors
  périmètre du jeu sont dans ce cas. Ces limites sont des cas du jeu, pas des oublis.
- **Avec un modèle, la réponse cite le bon document, et peut omettre un fait.** Le
  modèle lit les quatre sources et répond à la question posée : 9 cas en échec sans
  lui réussissent, 9 autres échouent, le plus souvent pour un fait attendu absent de
  sa formulation. Au total 20 cas en échec, autant que sans modèle, mais pas les mêmes.
- **L'agent conduit par le modèle aboutit dans 15 à 17 dossiers sur 18**, selon
  l'exécution. Dans les autres, le plan fixe prend le relais et le dossier reçoit quand
  même sa proposition de ticket.
- **Ces chiffres valent pour une famille de modèles.** Le juge de fidélité en fait partie
  et n'a pas été comparé à un jugement humain. Deux exécutions identiques ne donnent pas
  exactement les mêmes réponses, même à température 0.

Détail, méthode, cas en échec et limites : [docs/05-evaluation.md](docs/05-evaluation.md).

Linux / macOS :

```bash
make test          # 1 346 tests, hors ligne
make eval          # évaluation BM25 sans LLM, avec seuils de non-régression (celle de la CI)
make eval-hybrid   # la même avec le modèle d'embeddings
make eval-llm      # la même avec les modèles locaux (Ollama), réponses rédigées et jugées
PYTHONPATH=backend python scripts/smoke_llm.py   # vérifie les appels LLM configurés
```

Windows (PowerShell) :

```powershell
.\.venv\Scripts\python -m pytest -q
$env:PYTHONPATH = "backend"; $env:LLM_ENABLED = "off"; $env:RETRIEVAL_MODE = "bm25"
.\.venv\Scripts\python -m app.eval.run_eval
.\.venv\Scripts\python scripts/smoke_llm.py
```

Chaque exécution de l'évaluation réécrit le rapport de son mode sous `evals/reports/`.

## Structure

```
backend/app/
  agents/      tri (triage.py), extraction client/montant, agent et ses outils
  retrieval/   BM25, embeddings, stockage des vecteurs, fusion, réponses citées (rag.py)
  ingestion/   lecture Markdown / PDF / e-mail, découpage, construction de l'index
  services/    orchestration d'une demande, automatisation, validation humaine, indicateurs
  core/        configuration, masquage, garde-fous, client LLM, prompts, journalisation
  db/          modèles et accès (tickets, actions, journal des demandes, avis)
  eval/        évaluation et juge de fidélité
frontend/      interface Streamlit et son client HTTP
data/          documents fictifs (sample_docs/) et référentiel clients (reference/)
evals/         cas de référence, seuils, rapports
scripts/       vérification LLM, export des avis négatifs, génération du PDF d'exemple
docs/          cadrage, règles, architecture, décisions, évaluation, sécurité, exploitation
```

## Documentation

1. [Cadrage](docs/01-cadrage.md) — le problème, les hypothèses, ce qui est inventé
2. [Règles de gestion](docs/02-regles-gestion.md) — chaque règle, son code, son test
3. [Architecture](docs/03-architecture.md)
4. [Décisions](docs/04-decisions.md) — dont celles de ne pas utiliser l'IA
5. [Évaluation](docs/05-evaluation.md)
6. [Sécurité et données personnelles](docs/06-securite-rgpd.md)
7. [Exploitation](docs/07-exploitation.md)
8. [Industrialisation](docs/08-industrialisation.md) — ce qui reste à faire
9. [Scénario de démonstration](docs/09-demonstration.md)

## Comment ce projet a été réalisé

Prototype personnel, développé avec Claude Code comme assistant de programmation, puis
passé par une revue indépendante dont les constats ont été corrigés ou inscrits comme
limites dans l'évaluation. Les appels à un modèle, d'abord testés avec un modèle simulé,
ont ensuite été exécutés pour de bon avec des modèles locaux : cette première exécution
a révélé deux défauts, corrigés depuis ([docs/05-evaluation.md](docs/05-evaluation.md)).
Les règles, l'architecture et les arbitrages sont décrits dans `docs/` ; les chiffres de
ce README sont reproduits par les commandes ci-dessus.

## In English

Novare DossierOps is a prototype that handles customer requests for a fictional heating
and air-conditioning maintenance company. Deterministic business rules triage each
request, which then takes the simplest route that fits: a rule with no AI at all, an
answer quoted from the company's documents with its sources, an agent that prepares a
case file and proposes a ticket for a person to approve, or a hand-over to a person.
Without a model it still answers, by quoting passages as they are. The results above are
measured on 136 invented cases, first without an LLM, then with answers written by two
open-weight models running locally (Qwen 3.5 through Ollama); hosted providers go
through the same code and were not called. The documentation is in French.

## Licence

[MIT](LICENSE).
