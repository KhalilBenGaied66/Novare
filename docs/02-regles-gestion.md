# 2. Règles de gestion

Les règles sont évaluées dans l'ordre du tableau ; la première qui s'applique décide de
la voie. Aucune n'utilise de modèle : le code est dans
[`backend/app/agents/triage.py`](../backend/app/agents/triage.py), les tests dans
`backend/tests/test_triage.py` et `test_triage_hardening.py`.

## Tri des demandes

| Ordre | Règle | Condition | Voie |
|---|---|---|---|
| 1 | RG-04 | Sujet sensible : résiliation ou fin de contrat, pénalités, contentieux (avocat, tribunal, justice, médiateur, mise en demeure…), données personnelles (RGPD, CNIL, effacement, droit d'accès…) | `human` |
| 2 | RG-03 | Contestation explicite d'une facture, montant strictement inférieur à 500 €, client présent dans le référentiel, et rien d'autre dans la demande (pas de panne) | `automation` |
| 3 | RG-03b | Facturation, montant supérieur ou égal à 500 € | `human` |
| 4 | RG-05 | Panne ou dysfonctionnement, ou action demandée sur un dossier (planifier, dépanner, créer un ticket, vérifier le contrat) | `agent` |
| 5 | DEFAULT | Tout le reste : question documentaire | `rag` |

Précisions :

- **RG-04 passe avant RG-03.** Une demande qui mentionne un avocat et conteste 120 € est
  transmise à un gestionnaire, pas traitée automatiquement.
- **RG-03 exige une contestation dite en toutes lettres** (« je conteste », « erreur de
  facturation », « facturé deux fois », « demande de remboursement », « un avoir »…).
  Mentionner une facture ne suffit pas : « Merci pour la facture de 120 € » ne crée pas
  de ticket. C'est la seule règle qui agit sans validation, d'où cette exigence.
- **Client et montant** viennent des champs de la demande s'ils sont renseignés, sinon du
  texte (`C-12`, `120 €`, `1 250,50 euros`). Si le texte contient deux montants
  différents, aucun n'est retenu : rien ne dit lequel est contesté. Une quantité n'est
  pas multipliée (« 2 factures de 300 € » vaut 300 €).
- **Un client écrit dans le texte est un client déclaré, pas vérifié.** Il suffit pour
  ouvrir un ticket de litige (RG-03), pas pour consulter son contrat : seul le champ
  client donne accès aux documents et aux données d'un client.
- **Une question sans client identifié reste documentaire**, même avec le vocabulaire
  d'une panne (« Que faire en cas de panne ? »).
- Le texte est ramené à une forme unique avant le tri (accents composés, caractères
  invisibles retirés).
- Le vocabulaire est comparé sur des radicaux de mots (par préfixe), des mots entiers et
  des suites de mots, jamais sur des morceaux de mots : « ticket » ne contient pas « et ».
- Le seuil de RG-03 est le paramètre `RG03_MAX_AMOUNT`.

## Ce que les règles ne savent pas faire

Les règles lisent du vocabulaire, pas une intention. Les listes penchent du côté de la
personne : une fausse alerte de RG-04 ou RG-03b ne coûte qu'un regard de gestionnaire.
À l'inverse :

- un sujet sensible dit avec d'autres mots que ceux des listes n'est pas reconnu
  (« Nous comptons arrêter notre contrat » part en question documentaire) ;
- une panne décrite sans les termes prévus (« l'eau reste tiède ») n'est pas confiée à
  l'agent ;
- une demande d'un client identifié qui contient « panne » ou « planifier » va à l'agent
  même si c'est une simple question ; la proposition de ticket est alors à refuser.

Ces cas figurent dans le jeu d'évaluation et y échouent : voir
[05-evaluation.md](05-evaluation.md).

## Traitement par voie

| Voie | Règle | Code |
|---|---|---|
| `automation` | Crée un ticket « litige standard » sans validation. Une même demande du même client dans les 7 jours réutilise le ticket existant. | `services/automation.py` |
| `rag` | Réponse à partir des documents. Voir les garde-fous ci-dessous. | `retrieval/rag.py` |
| `agent` | Consulte les documents, lit le contrat du client, calcule l'échéance, rédige un brouillon et **propose** un ticket avec une priorité. | `agents/dossier_agent.py`, `agents/tools.py` |
| `human` | Aucune réponse automatique. La demande est enregistrée dans le journal des demandes et comptée dans `GET /api/v1/metrics` ; le prototype n'a ni file de traitement, ni affectation, ni notification (voir [08-industrialisation.md](08-industrialisation.md)). | `services/ask.py` |

## Garde-fous

| Règle | Comportement | Test |
|---|---|---|
| Pas de réponse sans source | Une réponse documentaire sans citation valide n'est jamais renvoyée : la demande est transmise. | `test_rag.py` |
| Sources insuffisantes | Confiance de recherche sous le seuil (`MIN_CONFIDENCE`, 0,35), ou aucun passage ne partage au moins deux termes avec la question : transmission. | `test_rag.py`, `test_api.py` |
| Validation humaine | Un ticket d'intervention n'existe qu'après approbation ; le valideur confirme ou corrige la priorité proposée. Une décision est définitive ; la répéter ne crée pas de second ticket. | `test_actions.py`, `test_api.py`, `test_hardening.py` |
| Cloisonnement | Un contrat n'est recherché, cité ou lu que pour le client du champ client. Les outils de l'agent n'acceptent pas d'identifiant client. | `test_retrieval.py`, `test_tools.py`, `test_hardening.py` |
| Données personnelles | Courriels, téléphones, IBAN, cartes, numéros de sécurité sociale et SIRET sont masqués avant stockage et avant tout appel à un modèle, sur toutes les voies. | `test_pii.py`, `test_api.py`, `test_hardening.py` |
| Réponse tronquée | Une réponse de modèle coupée à la limite de tokens n'est pas renvoyée : repli sur les extraits ou sur le plan fixe. | `test_hardening.py` |
| RG-08, budget | Un coût LLM supérieur à `MAX_COST_EUR_PER_REQUEST` (0,02 €) est signalé dans le journal de décision. La demande n'est pas interrompue. Le coût vient de la table de prix de LiteLLM : un modèle qui n'y figure pas est compté à 0 € et la règle ne se déclenche pas. | `test_rag.py`, `test_agent.py` |
| Erreur interne | Une erreur dans une voie devient une transmission à un gestionnaire, sans détail technique dans la réponse. | `test_api.py`, `test_ask_service.py` |

## Données de référence

Les délais et les contrats ne sont pas dans le code :

- `data/reference/clients.json` : clients, formule, dates de validité, astreinte, et
  délais d'intervention en heures par formule et par priorité ;
- `data/sample_docs/` : procédures, grille tarifaire, conditions générales, contrats.

Les délais se comptent en heures ouvrées (lundi à vendredi, 8 h à 18 h, heure de Paris),
sauf pour les clients dont le contrat inclut l'astreinte. Les jours fériés ne sont pas
gérés.
