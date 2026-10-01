# 5. Évaluation

## Méthode

`PYTHONPATH=backend python -m app.eval.run_eval` (ou `make eval`, `make eval-hybrid`)
fait passer chaque cas de `evals/golden_set.json` par le pipeline réel (`triage` puis
`handle_ask`, comme une requête d'API), sur un index reconstruit à partir du corpus et une
base de données temporaire. Un rapport par configuration est écrit dans `evals/reports/`
(`hybrid`, `bm25`).

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

**Les réponses rédigées par un modèle ne sont pas mesurées.** Aucune clé n'était
disponible. Avec une clé : `python -m app.eval.run_eval --judge` ajoute un contrôle de
fidélité par un second modèle (testé ici avec un modèle simulé uniquement).

## Cas en échec

20 cas en recherche hybride, 21 en BM25 (G-034 en plus).

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
- Pas de mesure du mode LLM, ni de l'agent piloté par un modèle.
- Les faits attendus sont vérifiés par expressions régulières : une réponse juste
  formulée autrement serait comptée fausse, et une expression trop large peut se
  trouver dans une phrase qui ne répond pas (le cas G-081 a été resserré pour cette
  raison).
- Pour les transmissions, seule la voie est contrôlée.

## Boucle d'amélioration

Un avis KO laissé dans l'interface est exporté par `scripts/feedback_to_golden.py`
comme cas candidat (question masquée, voie observée, commentaire). Une personne complète
les attendus avant de l'ajouter au jeu de référence. Rien n'est ajouté automatiquement.
