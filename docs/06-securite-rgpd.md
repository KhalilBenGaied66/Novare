# 6. Sécurité et données personnelles

Ce document dit ce qui est en place dans le code, ce qui ne l'est pas, et ce qui n'a
pas pu être vérifié. Il ne remplace pas une analyse d'impact (AIPD) ni un audit.

## Où va le texte d'une demande

| Étape | Ce qui est utilisé | Stocké ? |
|---|---|---|
| Réception (`POST /api/v1/ask`) | Texte brut, dans le corps de la requête | Non |
| Tri par règles | Texte brut, en mémoire (pour lire le montant et le client) | Non |
| Masquage | Remplace les identifiants par `[TEL]`, `[IBAN]`… | — |
| Recherche documentaire | Texte masqué, sans les balises | Non |
| Appel à un modèle (si activé) | Texte masqué et extraits de documents | Chez le fournisseur, selon son contrat |
| Journal des demandes | Texte masqué, types de données masquées | Oui |
| Ticket, proposition d'action | Texte masqué | Oui |
| Avis, motif de refus | Commentaire masqué | Oui |
| Journaux techniques | Identifiant de requête, voie, durées, longueur du texte ; jamais le texte | Sortie d'erreur du processus |

Aucune route `GET` ne reçoit de texte libre : une demande ne peut pas se retrouver dans
une URL ni dans un journal d'accès. Le journal d'accès n'écrit que le chemin, sans
paramètres.

## Masquage

`backend/app/core/pii.py`, testé par `backend/tests/test_pii.py`.

| Type | Reconnu | Contrôle |
|---|---|---|
| Courriel | Adresses usuelles | Forme |
| Téléphone | Numéros français : `06 12 34 56 78`, `0612345678`, `06.12.34.56.78`, `+33 6…`, `0033 6…`, `+33 (0)6…`, fixes 01 à 05 et 09 | Forme |
| IBAN | Tous pays, avec ou sans espaces | Clé modulo 97 |
| Carte bancaire | 13 à 19 chiffres | Clé de Luhn |
| Numéro de sécurité sociale | 15 chiffres, structure française | Structure (clé non vérifiée) |
| SIRET | 14 chiffres | Clé de Luhn |

**Ce que le masquage ne fait pas :**

- **Noms et adresses postales** ne sont pas reconnus. Une demande signée « Jean Dupont,
  12 rue de la Paix » est stockée et envoyée au modèle telle quelle. Il faudrait un
  modèle de reconnaissance d'entités.
- Les numéros de téléphone étrangers et les numéros en 08 ne sont pas masqués.
- La clé de Luhn accepte un nombre sur dix au hasard : une longue référence chiffrée
  peut être masquée à tort.
- Le contenu des documents du corpus n'est pas masqué. Le corpus fourni ne contient
  aucune donnée personnelle ; un corpus réel devrait être contrôlé avant ingestion.

## Cloisonnement entre clients

Un document marqué `client_id` (un contrat) n'est retrouvé, cité ou transmis à un
modèle que pour une demande de ce client. Le filtre est appliqué dans la recherche BM25
et dans les deux stockages de vecteurs ; les outils de l'agent n'acceptent pas
d'identifiant client. L'évaluation compte zéro document interdit atteint sur quatre cas
dédiés.

**Limite importante :** l'identité du client vient du champ `client_id` de la requête ou
du texte ; elle n'est pas authentifiée. Le cloisonnement empêche qu'une réponse préparée
pour un client contienne le contrat d'un autre. Il ne protège pas contre un appelant de
l'API qui indiquerait volontairement un autre identifiant : cela relève de
l'authentification des personnes, qui n'est pas faite.

## Accès à l'API

| En place | Non fait |
|---|---|
| Clé d'API unique (`X-API-Key`), comparaison à temps constant | Comptes individuels, SSO, rôles |
| Limite de débit par clé ou adresse (en mémoire, par processus) | Limite partagée entre plusieurs instances |
| Validation des entrées (longueur, format du client, montant positif) | — |
| Erreurs sans détail technique dans les réponses | — |
| CORS désactivé sauf configuration explicite | — |

Sans `API_KEY`, l'API est ouverte : c'est le réglage par défaut, prévu pour le
développement local. Le nom du valideur d'un ticket est un texte libre, pas une
identité vérifiée.

## Appels à un modèle de langage

- Avec `LLM_ENABLED=off`, ou sans clé de fournisseur, rien ne sort de la machine.
- Sinon, le fournisseur reçoit la question masquée et les extraits de documents
  retenus. Le choix du fournisseur (localisation, durée de conservation, usage pour
  l'entraînement) est une décision contractuelle qui reste à prendre ; le modèle par
  défaut des réponses documentaires est chez un fournisseur européen.
- Le modèle d'embeddings tourne localement : les documents ne sont envoyés à personne
  pour l'indexation.

## Injection d'instructions

Une demande ou un document peut contenir un texte qui cherche à donner des ordres au
modèle. Les protections sont structurelles :

- le tri ne dépend d'aucun modèle : une demande ne peut pas se faire router ailleurs
  en le demandant ;
- les outils de l'agent ne peuvent ni changer de client ni créer un ticket ;
- toute action passe par une validation humaine ;
- une réponse documentaire sans citation valide est rejetée ;
- les prompts présentent la demande et les sources comme des données, entre balises.

Il n'y a pas de détection d'injection, et la résistance des prompts n'a pas été
éprouvée contre un vrai modèle.

## Non fait

- Durée de conservation et purge des demandes, tickets et avis.
- Chiffrement applicatif des données stockées (à assurer par la base ou le disque).
- Coffre de secrets : les clés sont lues dans l'environnement ou `.env`.
- Journal d'audit des consultations (qui a lu quel ticket).
- Analyse d'impact, registre des traitements, procédure d'exercice des droits. Les
  demandes qui mentionnent le RGPD ou les données personnelles sont transmises à un
  gestionnaire, sans traitement automatique.
- Analyse de vulnérabilités des dépendances et des images.

## Non vérifié

- Comportement réel des fournisseurs de modèles (aucune clé pendant le développement).
- Configuration Docker : utilisateur non privilégié, ports liés à `127.0.0.1`, mot de
  passe PostgreSQL exigé sans valeur par défaut. Écrite, jamais exécutée.
