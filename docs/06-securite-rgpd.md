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
| Appel à un modèle (si activé) | Texte masqué et extraits de documents ; sur la voie agent, aussi l'identifiant du client et les résultats des outils (résumé du contrat, échéance) | Chez le fournisseur, selon son contrat |
| Journal des demandes | Texte masqué, types de données masquées | Oui |
| Ticket, proposition d'action | Texte masqué (y compris quand il est rédigé par un modèle) | Oui |
| Avis, motif de refus | Commentaire masqué | Oui |
| Nom du valideur | Texte libre, non masqué | Oui (actions, tickets) |
| Journaux techniques | Identifiant de requête, voie, durées, longueur du texte ; jamais le texte | Sortie d'erreur du processus |

Aucune route `GET` ne reçoit de texte libre : une demande ne peut pas se retrouver dans
une URL ni dans un journal d'accès. Le journal d'accès n'écrit que le chemin, sans
paramètres. Une requête refusée à la validation (422) renvoie le champ en cause et le
motif, pas la valeur refusée.

## Masquage

`backend/app/core/pii.py`, testé par `backend/tests/test_pii.py`.

| Type | Reconnu | Contrôle |
|---|---|---|
| Courriel | Adresses usuelles | Forme |
| Téléphone | Numéros français, national ou `+33` / `0033`, fixes 01 à 05 et 09 : paires séparées par espace, point ou tiret, ou tout regroupement avec des espaces (`06 12 345 678`) | Forme |
| IBAN | Tous pays, avec ou sans espaces | Clé modulo 97 |
| Carte bancaire | 13 à 19 chiffres | Clé de Luhn |
| Numéro de sécurité sociale | 15 chiffres, structure française | Structure (clé non vérifiée) |
| SIRET | 14 chiffres | Clé de Luhn |

**Ce que le masquage ne fait pas :**

- **Noms et adresses postales** ne sont pas reconnus. Une demande signée « Jean Dupont,
  12 rue de la Paix » est stockée et envoyée au modèle telle quelle. Il faudrait un
  modèle de reconnaissance d'entités.
- Les numéros de téléphone étrangers et les numéros en 08 ne sont pas masqués, ni les
  numéros à séparateurs mélangés (`06-12 34 56 78`) ou regroupés avec des points ou des
  tirets autrement que par paires.
- Un IBAN, une carte ou un numéro de sécurité sociale séparé par des points ou des
  tirets n'est pas reconnu.
- La clé de Luhn accepte un nombre sur dix au hasard : une longue référence chiffrée
  peut être masquée à tort.
- Le contenu des documents du corpus n'est pas masqué. Le corpus fourni ne contient
  aucune donnée personnelle ; un corpus réel devrait être contrôlé avant ingestion.

## Cloisonnement entre clients

Un document marqué `client_id` (un contrat) n'est retrouvé, cité ou transmis à un
modèle que pour une demande dont le **champ client** désigne ce client. Un identifiant
simplement écrit dans le texte n'ouvre rien : ni le contrat dans la recherche, ni les
données du référentiel dans les outils de l'agent. Le filtre est appliqué dans la
recherche BM25 et dans les deux stockages de vecteurs ; les outils de l'agent
n'acceptent pas d'identifiant client. L'évaluation compte zéro document interdit atteint
sur cinq cas dédiés.

**Limite importante :** le champ client lui-même n'est pas authentifié. Le cloisonnement
empêche qu'une réponse préparée pour un client contienne le contrat d'un autre, et
qu'une demande anonyme obtienne un contrat en citant un identifiant. Il ne protège pas
contre un appelant de l'API qui renseignerait volontairement le champ avec un autre
identifiant : cela relève de l'authentification des personnes, qui n'est pas faite. De
même, un appelant peut ouvrir un ticket de litige (RG-03) au nom de n'importe quel
client connu ; seule la limite de débit le borne.

## Accès à l'API

| En place | Non fait |
|---|---|
| Clé d'API unique (`X-API-Key`), comparaison à temps constant | Comptes individuels, SSO, rôles |
| Limite de débit par clé ou adresse (en mémoire, par processus) | Limite partagée entre plusieurs instances ; limite sur les tentatives de clé invalide (non comptées) |
| Validation des entrées (longueur, format du client, montant positif) | — |
| Erreurs internes sans détail technique ; erreurs de validation sans la valeur refusée | — |
| CORS désactivé sauf configuration explicite | — |

Sans `API_KEY`, l'API est ouverte : c'est le réglage par défaut, prévu pour le
développement local. `/health`, `/ready` et la documentation interactive (`/docs`)
restent accessibles sans clé. Le nom du valideur d'un ticket est un texte libre, pas une
identité vérifiée.

## Appels à un modèle de langage

- Avec `LLM_ENABLED=off`, ou sans clé de fournisseur, rien ne sort de la machine.
- Sinon, le fournisseur reçoit la question masquée et les extraits de documents
  retenus. Sur la voie agent, il reçoit aussi l'identifiant du client et les résultats
  des outils : résumé du contrat tiré du référentiel (nom du client, formule, dates,
  sites couverts) et échéance calculée.
- Modèles par défaut : réponses documentaires chez Mistral (fournisseur européen), agent
  chez Anthropic (fournisseur américain). Le choix des fournisseurs (localisation, durée
  de conservation, usage pour l'entraînement) est une décision contractuelle qui reste à
  prendre.
- Le modèle d'embeddings tourne localement : les documents ne sont envoyés à personne
  pour l'indexation.

## Injection d'instructions

Une demande ou un document peut contenir un texte qui cherche à donner des ordres au
modèle. Les protections sont structurelles :

- le tri ne dépend d'aucun modèle : une demande ne peut pas se faire router ailleurs
  en le demandant ;
- les outils de l'agent ne peuvent ni changer de client ni créer un ticket ;
- toute action proposée par l'agent passe par une validation humaine (seule la règle
  RG-03, sans modèle, crée un ticket sans validation) ;
- une réponse documentaire sans citation valide est rejetée ;
- les prompts présentent la demande et les sources comme des données, entre balises, et
  ces balises sont retirées du texte de la demande et des documents avant l'envoi : une
  demande ne peut pas refermer son bloc pour en ouvrir un faux.

Il n'y a pas de détection d'injection, et la résistance des prompts n'a pas été
éprouvée contre un vrai modèle.

## Non fait

- Durée de conservation et purge des demandes, tickets et avis.
- Chiffrement applicatif des données stockées (à assurer par la base ou le disque).
- Coffre de secrets : les clés sont lues dans l'environnement ou `.env`.
- Journal d'audit des consultations (qui a lu quel ticket).
- Analyse d'impact, registre des traitements, procédure d'exercice des droits. Les
  demandes qui mentionnent le RGPD ou les données personnelles avec le vocabulaire prévu
  sont transmises à un gestionnaire ; une formulation inattendue ne l'est pas.
- Analyse de vulnérabilités des dépendances et des images.

## Non vérifié

- Comportement réel des fournisseurs de modèles hébergés (aucune clé pendant le
  développement ; seuls des modèles locaux ont été exécutés).
- Configuration Docker : utilisateur non privilégié, ports liés à `127.0.0.1`, mot de
  passe PostgreSQL exigé sans valeur par défaut (il ne doit contenir ni `@` ni `%`).
  Exécutée en CI seulement.
