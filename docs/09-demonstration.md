# 9. Scénario de démonstration

Cinq minutes, sans clé d'API. Lancer l'API et l'interface (voir le README), puis ouvrir
http://localhost:8501.

## 1. Une demande que l'IA n'a pas à traiter

Bouton **Litige de 120 €**, puis **Traiter la demande**.

- Voie : automatisation par règle métier. Un ticket est créé, sans modèle ni recherche.
- Ouvrir le journal de décision : client reconnu, montant lu, règle RG-03.
- Renvoyer la même demande : le ticket existant est réutilisé, pas de doublon.

## 2. Une question documentaire

Bouton **Tarif week-end**.

- Voie : réponse documentaire. La réponse cite les passages de la grille tarifaire
  (+35 %), avec la section et la page de chaque source.
- Le mode affiché est « extraits des documents, sans LLM » : rien n'est reformulé.

Variante à saisir : « Quelle est la recette de la tarte aux pommes ? » La demande est
transmise à un gestionnaire, sans extrait.

## 3. Un dossier qui demande une action

Bouton **Panne de chaudière**.

- Voie : agent. Le journal montre les étapes : recherche, lecture du contrat, calcul de
  l'échéance, proposition de ticket.
- Le brouillon reprend la formule du client, le délai contractuel et l'échéance en
  heures ouvrées.
- Aucun ticket n'existe à ce stade. Saisir un nom de valideur, confirmer ou corriger la
  priorité, puis cliquer sur **Valider la création du ticket** : le ticket est créé et
  le formulaire laisse place au résultat. Un second appel à l'API pour la même
  proposition renvoie le même ticket.

## 4. Une demande sensible

Saisir : « Je veux résilier mon contrat, mon avocat vous contactera. Facture de 120 €. »
avec le client `C-12`.

- Voie : transmission à un gestionnaire, alors que le montant et le client rempliraient
  la règle d'automatisation. La règle des sujets sensibles passe avant.

## 5. Le cloisonnement

Saisir : « Quelle est la franchise prévue au contrat du client C-34 ? » avec le client
`C-45`.

- Le contrat de C-34 n'apparaît ni dans la réponse ni dans les sources. Même résultat
  en laissant le champ client vide : citer un identifiant dans le texte n'ouvre pas le
  contrat. Avec le client `C-34`, son contrat figure parmi les sources citées.
- En mode extraits, le passage cité peut être sans rapport avec la question : c'est la
  limite du mode sans modèle, décrite dans [05-evaluation.md](05-evaluation.md).

## 6. Les chiffres

- Barre latérale : demandes par voie, part des transmissions, latence, coût LLM (0 €
  sans modèle).
- En ligne de commande : `make eval` affiche le rapport d'évaluation, cas en échec
  compris, et échoue si un seuil n'est pas tenu.

## Avec une clé d'API

Renseigner la clé dans `.env`, relancer l'API, puis vérifier avec
`PYTHONPATH=backend python scripts/smoke_llm.py`. Les mêmes demandes sont alors rédigées
par un modèle ; le mode, le modèle, les tokens et le coût s'affichent sous la réponse.
Ce parcours n'a pas été exécuté pendant le développement.
