# 1. Cadrage

## Statut de ce document

Novare Services n'existe pas. Ce cadrage décrit un cas d'étude construit pour le
prototype : une situation plausible dans une entreprise de maintenance multitechnique,
pas le résultat d'entretiens ou d'observations. Les chiffres cités ci-dessous sont des
hypothèses de travail, pas des mesures.

## Situation de départ (hypothèse)

Une équipe de gestionnaires reçoit par courriel les demandes de clients sous contrat de
maintenance : pannes, demandes d'intervention, questions sur les tarifs et les délais,
contestations de factures. Pour répondre, un gestionnaire doit :

1. retrouver la procédure ou le tarif applicable dans des documents dispersés ;
2. vérifier le contrat du client (formule, validité, astreinte) ;
3. appliquer la bonne règle (délai selon la formule et la priorité, majorations) ;
4. répondre, et le cas échéant créer un ticket.

## Irritants visés

| Irritant | Ce que le prototype y oppose |
|---|---|
| Temps passé à chercher la bonne procédure | Recherche dans les documents, réponse avec ses sources |
| Réponses envoyées sans référence, source de litiges | Aucune réponse documentaire sans citation |
| Délais et majorations mal appliqués | Délais calculés à partir du référentiel clients, tarifs cités depuis la grille |
| Plusieurs tickets pour une même demande | Détection de doublon sur les litiges traités automatiquement |
| Pas de retour sur la qualité des réponses | Avis OK / KO par demande, exportable vers le jeu d'évaluation |

## Ce que le système ne doit pas faire

- Répondre seul sur un sujet sensible (résiliation, pénalités, contentieux, données
  personnelles).
- Créer un ticket d'intervention sans validation d'un gestionnaire.
- Montrer le contrat d'un client à un autre.
- Annoncer un délai, un tarif ou une clause qui ne figure pas dans les documents.

## Indicateurs à suivre en situation réelle

Ces indicateurs sont produits par l'API (`GET /api/v1/metrics`) ; aucune valeur cible
n'a été validée avec des utilisateurs.

- part des demandes traitées sans LLM ;
- part des demandes transmises à un gestionnaire ;
- taux d'avis positifs, par voie ;
- latence et coût par demande ;
- actions en attente de validation.

## Questions ouvertes

Un cadrage réel devrait trancher ces points avec les équipes concernées :

- Quels sujets exactement doivent toujours revenir à une personne ?
- Le seuil de 500 € pour les litiges traités automatiquement est une hypothèse : quel
  est le bon, et qui le fixe ?
- À qui, et par quel canal, une demande transmise est-elle remise ?
- Qui valide les tickets proposés, et dans quel délai ?
- Où vivent réellement les documents et les contrats (GED, ERP, outil de tickets) ?
- Combien de temps conserve-t-on les demandes, et sous quelle forme ?
