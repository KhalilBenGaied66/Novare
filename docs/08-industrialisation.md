# 8. Industrialisation

Ce prototype montre un parcours complet sur des données fictives. Cette page liste ce
qu'il faudrait faire avant de le mettre entre les mains de gestionnaires, avec les
équipes concernées. Rien de ce qui suit n'est fait, sauf mention contraire.

## Avant tout : valider le besoin

| Sujet | Avec qui |
|---|---|
| Essai avec de vrais gestionnaires sur de vraies demandes (anonymisées) | Métier |
| Revue des règles de tri, des sujets sensibles et du seuil des litiges | Métier, juridique |
| Jeu d'évaluation construit à partir de courriers réels | Métier |
| Mesure du mode LLM (fidélité, coût, latence) sur ce jeu | Équipe IA |

Tant que ces points ne sont pas traités, les chiffres de l'évaluation ne disent rien de
l'usage réel.

## Sécurité et conformité

| Sujet | État | À faire |
|---|---|---|
| Identité des utilisateurs | Clé d'API unique | SSO d'entreprise, rôles (gestionnaire, valideur, administrateur) |
| Identité du client d'une demande | Champ libre | Rattachement à la boîte de réception ou au portail client |
| Secrets | Variables d'environnement | Coffre de secrets |
| Données personnelles | Masquage par expressions régulières | Reconnaissance des noms et adresses, analyse d'impact, durées de conservation, purge |
| Fournisseurs de modèles | Paramétrables | Contrats, localisation des traitements, clauses de non-réutilisation |
| Dépendances et images | Versions figées | Analyse de vulnérabilités dans la CI |

## Exploitation

| Sujet | État | À faire |
|---|---|---|
| Déploiement | Dockerfiles et compose écrits, non exécutés | Les exécuter, puis cible d'hébergement de l'entreprise |
| Base de données | SQLite testé ; PostgreSQL configuré, non exécuté | Migrations de schéma, sauvegardes |
| Vecteurs | Fichier local testé ; Qdrant écrit, testé en mémoire seulement | Serveur ou service géré, index sur le champ client |
| Montée en charge | Un processus, traitement synchrone | File de tâches pour l'agent, limite de débit partagée |
| Observabilité | Journaux JSON, indicateurs agrégés | Traces des appels LLM, tableaux de bord, alertes |
| CI | Écrite (tests, évaluation, images), jamais exécutée | Premier passage sur le dépôt |

## Intégrations

Le prototype lit des fichiers et écrit des tickets dans sa propre base. En situation
réelle :

- les documents viennent d'une GED ou d'un intranet, avec leurs droits d'accès ;
- les contrats et les clients viennent du système de gestion, pas d'un fichier JSON ;
- les tickets sont créés dans l'outil de tickets existant ;
- les demandes arrivent d'une boîte de réception ou d'un portail.

Les points de branchement sont isolés : `ingestion/loaders.py` pour les documents,
`domain/clients.py` pour le référentiel, `db/repositories.py` pour les tickets,
`retrieval/vector_store.py` pour les vecteurs.

## Qualité des réponses

- Mesurer les réponses rédigées par un modèle, puis décider si le mode extraits reste
  le repli ou devient un simple mode dégradé.
- Reclassement des passages (reranking) si la base documentaire grandit.
- Découpage adapté au modèle d'embeddings (il ne lit qu'environ 450 caractères).
- Jours fériés et fuseaux horaires dans le calcul des échéances.

## Ordre proposé

1. Essai métier sur données réelles anonymisées, avec un modèle activé.
2. Authentification et rattachement du client.
3. Exécution de la CI et des images ; PostgreSQL et migrations.
4. Intégration à l'outil de tickets, en lecture puis en écriture.
5. Conformité (analyse d'impact, conservation) avant toute ouverture à des clients.
