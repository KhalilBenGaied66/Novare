# Novare DossierOps

Assistant de traitement des demandes clients pour un gestionnaire de maintenance. Chaque
demande est d'abord triée par des règles métier, puis traitée par la voie la plus simple
qui suffit : une règle sans IA, une réponse documentaire citée, un agent qui prépare un
dossier, ou une transmission à une personne.

> **Contexte.** Novare Services est une entreprise fictive (maintenance chauffage,
> ventilation, climatisation). Les documents, les clients et le jeu d'évaluation sont
> inventés pour ce prototype. Aucun utilisateur réel ne l'a testé.

## Ce que fait le système

| Demande | Voie | Ce qui se passe | IA générative |
|---|---|---|---|
| « Je conteste la facture de 120 € » (client connu) | `automation` | Ticket standard créé, doublons détectés | aucune |
| « Quelle est la majoration le week-end ? » | `rag` | Réponse tirée des documents, avec ses sources | facultative |
| « La chaudière est en panne » (client connu) | `agent` | Contrat vérifié, délai calculé, brouillon de réponse, ticket **proposé** | facultative |
| « Je veux résilier, mon avocat vous écrira » | `human` | Transmission à un gestionnaire, sans réponse automatique | aucune |

Trois principes tiennent l'ensemble :

- **L'IA n'est pas la réponse par défaut.** Le tri est déterministe (aucun modèle) : la
  même demande prend toujours la même voie, et la règle appliquée est écrite dans le
  journal de décision.
- **Dans le doute, une personne décide.** Sujet sensible, sources insuffisantes, réponse
  sans citation, erreur interne : la demande est transmise. L'agent ne crée jamais de
  ticket ; il le propose, et un gestionnaire le valide.
- **Sans clé d'API, le système répond quand même.** Les réponses documentaires citent
  alors les passages tels quels et l'agent suit un plan fixe. Avec une clé, un modèle
  rédige, sous les mêmes contrôles.

## État du projet

| | |
|---|---|
| **Fonctionne et testé** | Tri par règles, automatisation des petits litiges, recherche hybride (BM25 + vecteurs), réponses citées, agent à outils (LangGraph) avec validation humaine, masquage des données personnelles, cloisonnement des contrats par client, API (FastAPI), base de données, interface (Streamlit), évaluation sur 111 cas |
| **Écrit, testé avec un modèle simulé** | Appels LLM (rédaction des réponses, agent qui choisit ses outils, juge de fidélité). Aucune clé n'était disponible pendant le développement : `scripts/smoke_llm.py` sert à le vérifier avec une vraie clé. |
| **Écrit, jamais exécuté** | Images Docker et `docker-compose.yml`, PostgreSQL, serveur Qdrant, pipeline GitHub Actions. Sur la machine de développement, tout tourne avec SQLite et un index de vecteurs local. |
| **Non fait** | Authentification des personnes et rôles, migrations de schéma, purge automatique des données, traces LLM, déploiement cloud, test avec de vrais utilisateurs. Voir [docs/08-industrialisation.md](docs/08-industrialisation.md). |

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
.\.venv\Scripts\python -m streamlit run frontend/streamlit_app.py   # second terminal
```

Au premier démarrage, l'API construit l'index des documents et télécharge le modèle
d'embeddings (environ 220 Mo, dans `data/models/`). Pour s'en passer : `RETRIEVAL_MODE=bm25`.

Une demande en ligne de commande :

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"q": "Quelle est la majoration pour un deplacement le week-end ?"}'
```

Pour utiliser un modèle, copier `.env.example` en `.env` et renseigner la clé du
fournisseur (`MISTRAL_API_KEY` pour les réponses documentaires, `ANTHROPIC_API_KEY` pour
l'agent, ou d'autres modèles via `RAG_MODEL` et `AGENT_MODEL`), puis vérifier :

```bash
PYTHONPATH=backend python scripts/smoke_llm.py
```

Avec Docker (non vérifié, voir l'état du projet) : `POSTGRES_PASSWORD=... docker compose up --build`.

## Résultats mesurés

Évaluation du pipeline réel sur 111 cas (`evals/golden_set.json`), sans LLM, le
1er octobre 2026. Rapport complet : [evals/reports/latest.md](evals/reports/latest.md).

| Mesure | Recherche hybride | BM25 seul |
|---|---|---|
| Voie choisie par le tri | 100 % | 100 % |
| Voie finale (après garde-fous) | 98,2 % | 98,2 % |
| Document attendu parmi les 4 sources | 100 % | 100 % |
| Rang du document attendu (MRR) | 0,93 | 0,88 |
| Faits attendus présents dans la réponse | 93,1 % | 93,1 % |
| Document d'un autre client atteint | 0 | 0 |
| Latence moyenne | 35 ms | 2 ms |

Trois choses à savoir pour lire ces chiffres :

- Le jeu de cas a été écrit à partir des mêmes règles que le tri. Les 100 % de routage
  mesurent la conformité du code à ces règles, pas sa tenue face à de vrais courriers.
- Les 4 cas en échec sont des paraphrases (« Combien coûte la venue d'un technicien un
  samedi ? »). Sans LLM, la réponse cite des passages par recoupement de mots : elle ne
  reformule pas. Deux de ces cas sont transmis à un gestionnaire, deux citent un passage
  qui ne répond pas.
- La qualité des réponses rédigées par un modèle n'est pas mesurée ici.

Détail, méthode et limites : [docs/05-evaluation.md](docs/05-evaluation.md).

```bash
make test    # 1 216 tests, hors ligne
make eval    # évaluation BM25 sans LLM, avec seuils de non-régression (celle de la CI)
```

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
evals/         cas de référence, seuils, dernier rapport
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

Prototype personnel, développé avec Claude Code comme assistant de programmation. Les
règles, l'architecture et les arbitrages sont décrits dans `docs/` ; chaque chiffre de ce
README est reproduit par une commande indiquée ci-dessus.
