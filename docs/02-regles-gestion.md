# 2. Règles de gestion

Les règles sont évaluées dans l'ordre du tableau ; la première qui s'applique décide de
la voie. Aucune n'utilise de modèle : le code est dans
[`backend/app/agents/triage.py`](../backend/app/agents/triage.py), les tests dans
`backend/tests/test_triage.py`.

## Tri des demandes

| Ordre | Règle | Condition | Voie |
|---|---|---|---|
| 1 | RG-04 | Sujet sensible : résiliation, pénalités, contentieux (avocat, tribunal, huissier, mise en demeure…), données personnelles (RGPD, CNIL, droit d'accès…) | `human` |
| 2 | RG-03 | Litige de facturation, montant strictement inférieur à 500 €, client présent dans le référentiel | `automation` |
| 3 | RG-03b | Litige de facturation, montant supérieur ou égal à 500 € | `human` |
| 4 | RG-05 | Panne ou dysfonctionnement, ou action demandée sur un dossier (planifier, dépanner, créer un ticket, vérifier le contrat) | `agent` |
| 5 | DEFAULT | Tout le reste : question documentaire | `rag` |

Précisions :

- **RG-04 passe avant RG-03.** Une demande qui mentionne un avocat et conteste 120 € est
  transmise à un gestionnaire, pas traitée automatiquement.
- **Client et montant** viennent des champs de la demande s'ils sont renseignés, sinon du
  texte (`C-12`, `120 €`, `1 250,50 euros`). Si le texte contient deux montants
  différents, aucun n'est retenu : rien ne dit lequel est contesté.
- **Un litige sans montant, ou d'un client inconnu, n'est pas automatisé.** Il continue
  vers les règles suivantes.
- Le vocabulaire est comparé sur des radicaux de mots et des mots entiers, jamais sur
  des morceaux de mots : « ticket » ne contient pas « et ».
- Le seuil de RG-03 est le paramètre `RG03_MAX_AMOUNT`.

## Traitement par voie

| Voie | Règle | Code |
|---|---|---|
| `automation` | Crée un ticket « litige standard » sans validation. Une même demande du même client dans les 7 jours réutilise le ticket existant. | `services/automation.py` |
| `rag` | Réponse à partir des documents. Voir les garde-fous ci-dessous. | `retrieval/rag.py` |
| `agent` | Consulte les documents, lit le contrat du client, calcule l'échéance, rédige un brouillon et **propose** un ticket. | `agents/dossier_agent.py`, `agents/tools.py` |
| `human` | Aucune réponse automatique ; la demande est enregistrée et signalée. | `services/ask.py` |

## Garde-fous

| Règle | Comportement | Test |
|---|---|---|
| Pas de réponse sans source | Une réponse documentaire sans citation valide n'est jamais renvoyée : la demande est transmise. | `test_rag.py` |
| Sources insuffisantes | Confiance de recherche sous le seuil (`MIN_CONFIDENCE`, 0,35), ou aucun passage ne partage au moins deux termes avec la question : transmission. | `test_rag.py`, `test_api.py` |
| Validation humaine | Un ticket d'intervention n'existe qu'après approbation. Une décision est définitive ; la répéter ne crée pas de second ticket. | `test_actions.py`, `test_api.py` |
| Cloisonnement | Un contrat n'est recherché et cité que pour son propre client. Les outils de l'agent n'acceptent pas d'identifiant client. | `test_retrieval.py`, `test_tools.py` |
| Données personnelles | Courriels, téléphones, IBAN, cartes, numéros de sécurité sociale et SIRET sont masqués avant stockage et avant tout appel à un modèle. | `test_pii.py`, `test_api.py` |
| RG-08, budget | Un coût LLM supérieur à `MAX_COST_EUR_PER_REQUEST` (0,02 €) est signalé dans le journal de décision. La demande n'est pas interrompue. | `test_rag.py`, `test_agent.py` |
| Erreur interne | Une erreur dans une voie devient une transmission à un gestionnaire, sans détail technique dans la réponse. | `test_api.py`, `test_ask_service.py` |

## Données de référence

Les délais et les contrats ne sont pas dans le code :

- `data/reference/clients.json` : clients, formule, dates de validité, astreinte, et
  délais d'intervention en heures par formule et par priorité ;
- `data/sample_docs/` : procédures, grille tarifaire, conditions générales, contrats.

Les délais se comptent en heures ouvrées (lundi à vendredi, 8 h à 18 h), sauf pour les
clients dont le contrat inclut l'astreinte. Les jours fériés ne sont pas gérés.
