# 5. Évaluation

## Méthode

`python -m app.eval.run_eval` fait passer chaque cas de `evals/golden_set.json` par le
pipeline réel (`triage` puis `handle_ask`, comme une requête d'API), sur le corpus
configuré et une base de données temporaire. Le rapport est écrit dans
`evals/reports/latest.md` et `latest.json`.

Chaque cas porte ce qui est attendu :

| Champ | Contrôle |
|---|---|
| `expected_route` | Voie choisie par le tri |
| `expected_final_route` | Voie de la réponse (une question documentaire peut finir en transmission) |
| `expected_docs` | Au moins un de ces documents parmi les 4 sources, et cité dans la réponse |
| `forbidden_docs` | Aucun de ces documents ne doit être retrouvé ni cité (cloisonnement) |
| `expected_facts` | Expressions qui doivent toutes figurer dans la réponse |
| `forbidden_facts` | Expressions qui ne doivent pas y figurer |
| `expected_pii` | Types de données personnelles qui doivent être masqués |

## Jeu de cas

111 cas, répartis en `dev` (67, utilisé pour les réglages) et `test` (44, seulement mesuré).

| Catégorie | Cas | Contenu |
|---|---|---|
| Question documentaire | 48 | Tous les documents publics, dont le PDF et l'e-mail ; paraphrases |
| Hors périmètre | 17 | Questions sans rapport, dont 11 partageant un mot courant avec le corpus |
| Sujet sensible | 14 | Formes sans accent et conjuguées, litige de 500 € ou plus, demande RGPD |
| Dossier (agent) | 13 | Pannes, planification, client inconnu, contrat échu |
| Automatisation | 9 | Montant ou client dans le texte seulement, doublon |
| Données personnelles | 6 | Téléphone, courriel, IBAN, carte, numéro de sécurité sociale |
| Cloisonnement | 4 | Un client ou un anonyme demande le contrat d'un autre |

## Résultats

Sans LLM, le 1er octobre 2026. Colonnes : tous les cas / dev / test.

| Mesure | Recherche hybride | BM25 seul |
|---|---|---|
| Voie choisie par le tri | 1,00 / 1,00 / 1,00 | 1,00 / 1,00 / 1,00 |
| Voie finale | 0,982 / 0,970 / 1,00 | 0,982 / 0,970 / 1,00 |
| Document attendu parmi les sources | 1,00 / 1,00 / 1,00 | 1,00 / 1,00 / 1,00 |
| MRR | 0,927 / 0,969 / 0,855 | 0,885 / 0,938 / 0,794 |
| Document attendu cité | 0,967 / 0,974 / 0,957 | 0,967 / 0,974 / 0,957 |
| Faits attendus dans la réponse | 0,931 / 0,919 / 0,952 | 0,931 / 0,919 / 0,952 |
| Données personnelles détectées | 1,00 / 1,00 / 1,00 | 1,00 / 1,00 / 1,00 |
| Documents interdits atteints | 0 | 0 |
| Faits interdits dans la réponse | 0 | 0 |
| Latence moyenne / p95 | 35 ms / 71 ms | 2 ms / 7 ms |

## Comment lire ces chiffres

**Le routage à 100 % ne dit pas que le tri est juste sur de vrais courriers.** Les cas
ont été écrits à partir des règles de [02-regles-gestion.md](02-regles-gestion.md), comme
le code. Le chiffre mesure la conformité de l'un à l'autre et protège contre une
régression ; il ne mesure pas la couverture du vocabulaire réel des clients.

**Les deux modes de recherche donnent les mêmes réponses.** Les vecteurs améliorent le
rang du bon document (MRR), mais en mode extraits la réponse dépend du recoupement de
mots entre la question et les passages. L'apport des vecteurs sur la réponse finale ne
se verra qu'avec un modèle qui reformule.

**Les réponses rédigées par un modèle ne sont pas mesurées.** Aucune clé n'était
disponible. Avec une clé : `python -m app.eval.run_eval --judge` ajoute un contrôle de
fidélité par un second modèle (testé ici avec un modèle simulé uniquement).

## Cas en échec

Les quatre échecs sont des paraphrases, dans les deux modes :

| Cas | Question | Résultat |
|---|---|---|
| G-078 (dev) | « Combien coûte la venue d'un technicien un samedi ? » | Transmis à un gestionnaire |
| G-082 (dev) | « Puis-je décaler une visite au dernier moment sans rien payer ? » | Transmis à un gestionnaire |
| G-080 (dev) | « Combien de temps ai-je pour signaler une erreur sur une somme qui m'a été réclamée ? » | Cite un passage qui ne répond pas |
| G-079 (test) | « Notre contrat de maintenance est terminé : pouvez-vous encore intervenir chez nous ? » | Cite un passage qui ne répond pas |

Dans les quatre cas le bon document est parmi les sources ; c'est le choix du passage à
citer qui échoue, faute de mots communs.

## Réglages, et sur quoi ils ont été faits

**Seuil de confiance (0,35).** Sur le jeu `dev` en recherche hybride, la confiance la plus
basse d'une question dans le périmètre est 0,46 (0,48 sur `test`). Les questions hors
périmètre sans mot commun avec le corpus restent sous 0,25. Celles qui partagent un mot
courant montent jusqu'à 0,49 : le seuil seul ne les arrête pas, c'est l'objet de la
règle suivante.

**Règle des deux termes partagés.** Sept questions hors sujet essayées à la main
(« Quel temps fera-t-il demain à Lyon ? ») ont montré que le seuil de confiance ne
suffisait pas : trois recevaient un extrait sans rapport. Quatre variantes ont été
comparées, au moment du réglage, sur les questions documentaires du jeu et ces essais :

| Variante | Questions du périmètre répondues | Hors sujet répondues à tort |
|---|---|---|
| Un terme partagé suffit (avant) | 51 / 51 | 3 / 13 |
| Deux termes partagés (retenue) | 48 / 51 | 0 / 13 |
| Termes pondérés par leur rareté, deux termes | 47 à 48 / 51 | 0 / 13 |
| Termes pondérés par leur rareté, un terme | 49 à 51 / 51 | 3 / 13 |

La pondération par rareté n'apportait rien et a été retirée. Les sept questions d'essai
ont été ajoutées au jeu `dev` ; quatre autres, écrites après le réglage et non essayées
avant la mesure, ont été ajoutées au jeu `test` : les quatre sont transmises.

Un essai en conditions réelles a ensuite montré que la règle bloquait « Quels sont les
tarifs de nuit ? » (le passage dit « majoration de nuit », seul le titre du document parle
de tarifs). Le titre du document compte depuis comme contexte du passage ; le cas est au
jeu `dev` (G-110) et un cas de même forme, non essayé avant, au jeu `test` (G-111, réussi).

## Seuils de non-régression

`evals/thresholds.json` fixe un minimum par mesure, un peu sous les valeurs du mode le
plus faible, et zéro violation de cloisonnement. `make eval` et la CI échouent si un
seuil n'est pas tenu. Ce sont des garde-fous contre une régression, pas des objectifs.

## Limites de cette évaluation

- Corpus de 12 documents et 118 passages : rien n'est dit sur le comportement à
  l'échelle d'une vraie base documentaire.
- Cas écrits par l'auteur du système, sans courriers réels.
- Pas de mesure du mode LLM, ni de l'agent piloté par un modèle.
- Les faits attendus sont vérifiés par expressions régulières : une réponse juste
  formulée autrement serait comptée fausse.

## Boucle d'amélioration

Un avis KO laissé dans l'interface est exporté par `scripts/feedback_to_golden.py`
comme cas candidat (question masquée, voie observée, commentaire). Une personne complète
les attendus avant de l'ajouter au jeu de référence. Rien n'est ajouté automatiquement.
