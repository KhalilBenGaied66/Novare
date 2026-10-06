# 5. Évaluation

## Méthode

`PYTHONPATH=backend python -m app.eval.run_eval` (ou `make eval`, `make eval-hybrid`)
fait passer chaque cas de `evals/golden_set.json` par le pipeline réel (`triage` puis
`handle_ask`, comme une requête d'API), sur un index reconstruit à partir du corpus et une
base de données temporaire. Un rapport par configuration est écrit dans `evals/reports/`
(`hybrid`, `bm25`, et `hybrid-llm` quand un modèle rédige).

Chaque cas porte ce qui est attendu :

| Champ | Contrôle |
|---|---|
| `expected_route` | Voie choisie par le tri |
| `expected_final_route` | Voie de la réponse (une question documentaire peut finir en transmission) |
| `expected_docs` | Au moins un de ces documents parmi les 4 sources ; et, pour une demande qui doit recevoir une réponse, cité dans cette réponse (une transmission à tort compte comme un échec) |
| `forbidden_docs` | Aucun de ces documents ne doit être retrouvé ni cité (cloisonnement) |
| `expected_facts` | Expressions qui doivent toutes figurer dans la réponse |
| `forbidden_facts` | Expressions qui ne doivent pas y figurer |
| `expected_pii` | Types de données personnelles qui doivent être masqués |

## Jeu de cas

136 cas : `dev` (85), utilisé pour régler les règles et les seuils, et `test` (51).

| Catégorie | Cas | Contenu |
|---|---|---|
| Question documentaire | 56 | Tous les documents publics, dont le PDF et l'e-mail ; paraphrases ; trois limites connues |
| Hors périmètre | 22 | Sans rapport avec le corpus, ou partageant un ou deux mots du domaine |
| Sujet sensible | 20 | Formes sans accent, conjuguées, accents décomposés, autres formulations, litige de 500 € ou plus |
| Dossier (agent) | 17 | Pannes, planification, client cité dans le texte, contrat échu, panne et litige mêlés |
| Automatisation | 10 | Montant ou client dans le texte seulement, doublon, double facturation |
| Données personnelles | 6 | Téléphone, courriel, IBAN, carte, numéro de sécurité sociale |
| Cloisonnement | 5 | Un client, ou un anonyme qui cite un identifiant, demande le contrat d'un autre |

D'où viennent les cas, ce qui dit ce qu'ils mesurent :

- **Environ 100 cas écrits à partir des règles** de [02-regles-gestion.md](02-regles-gestion.md),
  comme le code. Ils mesurent la conformité de l'un à l'autre et protègent d'une
  régression.
- **Une trentaine de cas issus d'essais et d'une revue indépendante**, ajoutés au jeu
  `dev` : les questions qui ont mis les règles en défaut, corrigées depuis, et onze
  limites connues laissées en échec (étiquette `limite_connue`).
- **Huit cas écrits après les corrections, avec l'attendu métier, sans être essayés
  avant la mesure** (étiquette `non_essaye`, jeu `test`). C'est la seule partie du jeu
  qui renseigne sur des demandes que les règles n'ont pas vues.

## Résultats

Sans LLM. Colonnes : tous les cas / dev / test.

| Mesure | Recherche hybride | BM25 seul |
|---|---|---|
| Voie choisie par le tri | 0,985 / 1,00 / 0,961 | 0,985 / 1,00 / 0,961 |
| Voie finale | 0,904 / 0,882 / 0,941 | 0,904 / 0,882 / 0,941 |
| Document attendu parmi les sources (69 cas) | 1,00 / 1,00 / 1,00 | 0,986 / 1,00 / 0,958 |
| MRR | 0,889 / 0,920 / 0,830 | 0,861 / 0,915 / 0,760 |
| Document attendu cité (69 demandes à répondre) | 0,899 / 0,867 / 0,958 | 0,884 / 0,867 / 0,917 |
| Faits attendus dans la réponse (70 cas) | 0,871 / 0,864 / 0,885 | 0,871 / 0,864 / 0,885 |
| Données personnelles détectées | 1,00 / 1,00 / 1,00 | 1,00 / 1,00 / 1,00 |
| Documents interdits atteints | 0 | 0 |
| Faits interdits dans la réponse | 0 | 0 |

Les latences figurent dans les rapports à titre indicatif : elles dépendent de la machine
(quelques millisecondes en BM25, quelques dizaines avec le modèle d'embeddings).

## Comment lire ces chiffres

**Le routage du jeu `dev` à 100 % ne dit pas que le tri est juste sur de vrais courriers.**
Ces cas viennent des règles ou ont servi à les corriger. Sur les huit cas écrits ensuite
avec l'attendu métier, deux sont mal routés et deux autres échouent après le tri.

**Les deux modes de recherche obtiennent presque les mêmes scores, pas les mêmes
réponses.** Les vecteurs améliorent le rang du bon document (MRR) et retrouvent un
document que BM25 manque (G-034). En mode extraits, le choix des phrases citées dépend
du recoupement de mots entre la question et les passages : les passages cités diffèrent
souvent d'un mode à l'autre sans que les contrôles changent de résultat. L'apport des
vecteurs sur la réponse finale ne se verra qu'avec un modèle qui reformule.

**Les réponses rédigées par un modèle sont mesurées à part** : section suivante.

## Avec un modèle

`make eval-llm` refait la même évaluation en laissant des modèles rédiger. Ce sont deux
modèles ouverts, exécutés localement par Ollama sur une carte graphique de 16 Go :
Qwen 3.5 à 4 milliards de paramètres pour les réponses documentaires, Qwen 3.5 à
9 milliards pour l'agent et pour le juge de fidélité. Recherche hybride, température 0,
raisonnement désactivé, contexte de 16 384 tokens. Rapport :
[`evals/reports/hybrid-llm.md`](../evals/reports/hybrid-llm.md). Colonnes : tous les cas
/ dev / test.

| Mesure | Sans modèle | Avec les modèles locaux |
|---|---|---|
| Voie finale | 0,904 / 0,882 / 0,941 | 0,919 / 0,906 / 0,941 |
| Document attendu cité (69 demandes à répondre) | 0,899 / 0,867 / 0,958 | 1,00 / 1,00 / 1,00 |
| Faits attendus dans la réponse (70 cas) | 0,871 / 0,864 / 0,885 | 0,886 / 0,886 / 0,885 |
| Réponse jugée fidèle aux sources (70 réponses jugées) | — | 0,986 / 1,00 / 0,955 |
| Documents interdits atteints, faits interdits dans la réponse | 0 | 0 |
| Cas en échec | 20 | 19 |

Le tri et la recherche ne dépendent pas du modèle : leurs lignes sont celles du tableau
précédent.

**Ce que le modèle change.** Il cite un document attendu dans toutes les demandes à
répondre, là où le mode extraits en manquait 7 sur 69 : au lieu de retenir les phrases qui
partagent le plus de mots avec la question, il lit les quatre sources et répond à la
question posée. 9 cas en échec sans modèle réussissent (G-079, G-080, G-082, G-120, G-127,
G-128, G-129, G-130, G-134). En sens inverse, 8 cas qui réussissaient échouent (G-024,
G-025, G-034, G-037, G-055, G-069, G-104, G-113), le plus souvent parce qu'un fait attendu
manque dans sa formulation.

**L'agent conduit par le modèle aboutit dans 12 à 14 dossiers sur 18**, selon l'exécution.
Dans les autres, il n'a pas rendu de réponse finale dans les six tours autorisés, ou son
brouillon annonçait une proposition qu'il n'avait pas faite ; le plan fixe prend le
relais et le dossier reçoit quand même sa proposition de ticket.

**Hors périmètre.** Quand les sources ne répondent pas à la question, le modèle le dit
et la demande est transmise (6 cas). Il répond encore à 6 questions hors périmètre qui
partagent du vocabulaire avec un document.

**Temps de réponse.** 1,1 seconde pour une question documentaire, 19 secondes pour un
dossier (médianes, sur cette machine). Sans modèle : quelques dizaines de millisecondes.

**D'une exécution à l'autre.** Deux exécutions identiques ne donnent pas exactement les
mêmes réponses, même à température 0 : 19 cas en échec pour l'une et 20 pour l'autre
(G-033, G-034, G-044 diffèrent), et l'agent a conclu seul 14 dossiers dans la première, 12
dans la seconde. Le rapport déposé est celui de la seconde.

Les 19 cas en échec avec le modèle :

| Cas | Demande | Contrôle en échec |
|---|---|---|
| G-024 (dev) | « La chaudière de l'immeuble est en panne depuis ce matin, merci d'envoyer un technicien. » | faits absents de la réponse : ['8 h ouvrees', 'priorite p2'] |
| G-025 (dev) | « Fuite d'eau importante sur le réseau de chauffage du bloc technique, intervention urgente de… » | faits absents de la réponse : ['24 h/24'] |
| G-034 (test) | « Chaudière en panne sur le site du client C-27, les blocs opératoires ne sont plus chauffés. » | faits absents de la réponse : ['client non identifie'] |
| G-037 (dev) | « Quel est le délai d'intervention P2 pour la formule Confort ? » | faits absents de la réponse : ['8 h ouvrees'] |
| G-055 (dev) | « Le contrat de maintenance est-il renouvelé par tacite reconduction ? » | faits absents de la réponse : ['sans tacite reconduction'] |
| G-069 (test) | « L'astreinte du week-end est-elle incluse dans notre contrat ? » | faits absents de la réponse : ['astreinte incluse 24 h/24'] |
| G-078 (dev) | « Combien coûte la venue d'un technicien un samedi ? » | faits absents de la réponse : ['211,95'] |
| G-081 (test) | « Si j'appelle la nuit, au bout de combien de temps un technicien me recontacte-t-il ? » | faits absents de la réponse : ['rappelle le client dans les 20 minutes/rappel du client.*20 minutes'] |
| G-104 (dev) | « Vendez-vous des climatiseurs Daikin et à quel prix ? » | route finale rag, attendue human |
| G-113 (dev) | « Merci pour la facture de 120 €, bien reçue. » | route finale rag, attendue human |
| G-114 (dev) | « Je ne conteste pas la facture de 120 €, je demande seulement un duplicata. » | route finale rag, attendue human |
| G-122 (dev) | « Combien coûte un déplacement en taxi de Lyon à Vénissieux ? » | route finale rag, attendue human |
| G-123 (dev) | « Quel est le délai de livraison d'une pièce commandée sur Amazon ? » | route finale rag, attendue human |
| G-124 (dev) | « Quelle est la durée de garantie de mon lave-vaisselle ? » | route finale rag, attendue human |
| G-125 (dev) | « Quel est le tarif d'une nuit d'hôtel à Lyon le week-end ? » | route finale rag, attendue human |
| G-126 (dev) | « Quels sont les jours fériés en France en 2026 ? » | route finale rag, attendue human |
| G-131 (test) | « Nous comptons arrêter notre contrat à la fin de l'année. » | route rag, attendue human ; route finale rag, attendue human |
| G-132 (test) | « Le chauffe-eau fait un bruit de claquement et l'eau reste tiède. » | route rag, attendue agent ; route finale rag, attendue agent |
| G-133 (test) | « Suite à votre relance, je vous confirme le paiement de la facture de 310 €. » | route finale rag, attendue human ; réponse jugée non fidèle aux sources |

### Ce que la première exécution réelle a trouvé

Trois défauts que les tests, faits avec un modèle simulé, ne pouvaient pas montrer :

1. **Un outil sans argument faisait échouer la requête.** Quand Qwen 3.5 appelle
   `get_contract`, qui ne prenait aucun argument, il écrit un appel mal formé et le
   serveur répond à toute la requête par une erreur. Quatre dossiers sur dix-huit
   repassaient pour cela par le plan fixe. L'outil déclare désormais un paramètre
   facultatif, qu'il ignore ; un test vérifie qu'aucun outil n'est sans paramètre.
2. **Un modèle qui raisonne dépense sa réponse à raisonner.** Avec le raisonnement
   activé, qui est le réglage par défaut du serveur pour ce modèle, la réponse
   documentaire, limitée à 600 tokens, revenait tronquée et le garde-fou la remplaçait
   par des extraits.
3. **Un brouillon annonçait une proposition qui n'existait pas.** Dans quatre dossiers
   sur dix-huit, le modèle écrivait au client qu'une intervention était proposée et en
   attente de validation, sans avoir appelé `propose_ticket` : le gestionnaire aurait
   attendu une proposition jamais enregistrée. Un tel brouillon n'est plus retenu, et le
   plan fixe, qui propose le ticket, prend le relais.

Et une marge trop faible : la fenêtre de contexte par défaut du serveur est de
4 096 tokens, réponse comprise, quand l'historique d'un dossier atteint 3 000 tokens au
cinquième tour. Elle est portée à 16 384.

Le raisonnement et le contexte se règlent sans code propre à un fournisseur :
`LLM_EXTRA_PARAMS` ajoute à chaque appel les paramètres dont un modèle a besoin (voir
`.env.example`).

### Limites de cette mesure

- Une seule famille de modèles, et le juge en fait partie : il n'a pas été comparé à un
  jugement humain.
- Les fournisseurs hébergés (Mistral, Anthropic) passent par le même code et n'ont pas
  été appelés.
- Le jeu de cas a été écrit pour le mode sans modèle : des demandes en un seul message,
  dont la réponse tient le plus souvent dans un document.

## Cas en échec

Sans modèle : 20 cas en recherche hybride, 21 en BM25 (G-034 en plus).

**Paraphrases (5).** Le bon document est parmi les sources ; c'est le choix du passage à
citer qui échoue, faute de mots communs.

| Cas | Question | Résultat |
|---|---|---|
| G-078 (dev) | « Combien coûte la venue d'un technicien un samedi ? » | Transmis à un gestionnaire |
| G-082 (dev) | « Puis-je décaler une visite au dernier moment sans rien payer ? » | Transmis à un gestionnaire |
| G-080 (dev) | « Combien de temps ai-je pour signaler une erreur sur une somme qui m'a été réclamée ? » | Cite un passage qui ne répond pas |
| G-079 (test) | « Notre contrat de maintenance est terminé : pouvez-vous encore intervenir chez nous ? » | Cite un passage qui ne répond pas |
| G-081 (test) | « Si j'appelle la nuit, au bout de combien de temps un technicien me recontacte-t-il ? » | Cite un passage voisin, sans l'engagement de rappel |

**Limites connues du mode extraits (11, jeu `dev`).**

- Six questions hors sujet qui partagent deux termes du domaine avec un passage
  reçoivent un extrait sans rapport (G-122 à G-127 : « Combien coûte un déplacement en
  taxi de Lyon à Vénissieux ? » cite le tarif de déplacement).
- Trois questions du périmètre reçoivent un passage voisin mais faux (G-128 à G-130 :
  « délai de paiement des factures » cite le délai de contestation de 30 jours ; la
  réponse est 45 jours).
- Une demande qui n'est pas une question (« je demande seulement un duplicata », G-114)
  reçoit un extrait au lieu d'être transmise.
- « Que faire en cas de panne de chaudière ? » (G-120) est bien traitée comme une
  question, mais aucun passage ne la recoupe assez : elle est transmise.

**Cas non essayés avant la mesure (4 échecs sur 8, jeu `test`).**

| Cas | Demande | Résultat |
|---|---|---|
| G-131 | « Nous comptons arrêter notre contrat à la fin de l'année. » | Non reconnu comme une résiliation : question documentaire |
| G-132 | « Le chauffe-eau fait un bruit de claquement et l'eau reste tiède. » | Non reconnu comme une panne : transmis, sans dossier |
| G-133 | « … je vous confirme le paiement de la facture de 310 €. » | Pas de ticket (correct), mais un extrait est cité au lieu d'une transmission |
| G-134 | « Ma mère âgée n'a plus de chauffage depuis hier soir, c'est grave. » | Confié à l'agent (correct), priorité P2 au lieu de P1 |

Les quatre autres réussissent : double facturation automatisée, menace de justice
transmise, question hors sujet transmise, délai d'émission d'un avoir cité.

## Réglages, et sur quoi ils ont été faits

**Seuil de confiance (0,35).** Sur le jeu `dev` en recherche hybride, la confiance la plus
basse d'une question du périmètre est 0,46. Les questions hors périmètre sans mot commun
avec le corpus restent sous 0,25 ; celles qui partagent un mot courant montent jusqu'à
0,49, et plus haut encore avec deux. Le seuil seul ne les arrête pas, d'où la règle
suivante. En mode BM25 la confiance
est le seul recoupement de mots ; le seuil n'a pas été réglé séparément pour ce mode.

**Règle des deux termes partagés.** Sept questions hors sujet essayées à la main
(« Quel temps fera-t-il demain à Lyon ? ») recevaient pour trois d'entre elles un
extrait sans rapport. Exiger qu'un passage cité partage au moins deux termes avec la
question (en comptant le titre de sa section et celui du document) les fait toutes
transmettre, au prix de deux questions du périmètre désormais transmises (G-078, G-082).
La comparaison a porté sur les questions documentaires et hors périmètre des deux jeux :
le jeu `test` n'est donc pas neutre pour cette règle. La règle ne protège pas d'une
question hors sujet qui partage deux termes : ce sont les six limites ci-dessus.

**Corrections après revue indépendante.** Une revue du dépôt a montré que la règle
d'automatisation se déclenchait sur la simple mention d'une facture, que des menaces
juridiques dites autrement passaient, que des accents décomposés contournaient les
règles, qu'un montant écrit « 1 250 € » avec un séparateur inhabituel était lu 250 €, et
qu'un identifiant client écrit dans le texte ouvrait le contrat de ce client. Ces
défauts sont corrigés ; les demandes qui les ont révélés sont au jeu `dev`. La revue a
aussi relevé qu'un cas du jeu `test` recopiait un cas du jeu `dev` : il a été retiré, de
même qu'un cas `dev` qui paraphrasait un cas `test`.

## Seuils de non-régression

`evals/thresholds.json` fixe un minimum par mesure, un peu sous les valeurs du mode le
plus faible, et zéro violation de cloisonnement. `make eval` et la CI échouent si un
seuil n'est pas tenu. Ce sont des garde-fous contre une régression, pas des objectifs :
ils laissent passer une ou deux régressions isolées par mesure. Les comportements précis
(seuil des litiges, doublon, contrat échu, priorité) sont couverts par les tests
unitaires, pas par ces seuils.

## Limites de cette évaluation

- Corpus de 12 documents et 118 passages : rien n'est dit sur le comportement à
  l'échelle d'une vraie base documentaire.
- Cas écrits par l'auteur du système et par une revue, sans courriers réels.
- Le mode LLM et l'agent piloté par un modèle sont mesurés avec une seule famille de
  modèles locaux (voir « Avec un modèle »).
- Les faits attendus sont vérifiés par expressions régulières : une réponse juste
  formulée autrement serait comptée fausse, et une expression trop large peut se
  trouver dans une phrase qui ne répond pas (le cas G-081 a été resserré pour cette
  raison).
- Pour les transmissions, seule la voie est contrôlée.

## Boucle d'amélioration

Un avis KO laissé dans l'interface est exporté par `scripts/feedback_to_golden.py`
comme cas candidat (question masquée, voie observée, commentaire). Une personne complète
les attendus avant de l'ajouter au jeu de référence. Rien n'est ajouté automatiquement.
