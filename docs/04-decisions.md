# 4. Décisions

Chaque décision donne le choix retenu, ce qui a été écarté et ce que le choix coûte.

## D1. Le tri n'utilise pas de modèle

**Choix.** Des règles ordonnées sur le vocabulaire, le montant et le client.

**Pourquoi.** Le tri décide si une demande part chez un avocat ou devient un ticket
automatique. Il doit donner la même voie à la même demande, s'expliquer par une règle
nommée, et se tester sans réseau.

**Écarté.** Un classifieur LLM : non déterministe, payant à chaque demande, et son
erreur la plus grave (automatiser une demande sensible) ne se constate qu'après coup.

**Coût.** Les règles ne reconnaissent que le vocabulaire prévu. Une formulation
inattendue d'un sujet sensible passe en question documentaire ; elle reste soumise à
relecture, mais n'est pas signalée comme sensible.

## D2. Les petits litiges sont traités par une règle

**Choix.** Litige inférieur à 500 € d'un client connu : ticket standard, sans modèle
ni recherche documentaire.

**Pourquoi.** La décision ne dépend que de trois faits vérifiables. Un modèle
n'apporterait qu'un coût et une source d'erreur.

**Coût.** Le montant doit être lisible (champ ou texte sans ambiguïté). Deux montants
différents dans le texte bloquent l'automatisation.

## D3. Sans modèle, le système cite ; il n'invente pas

**Choix.** Sans clé d'API ou quand le modèle échoue, la réponse documentaire est faite
de passages cités tels quels, et l'agent suit un plan fixe.

**Pourquoi.** Le prototype doit pouvoir être essayé et testé sans compte chez un
fournisseur, et rester utile si le fournisseur est indisponible.

**Écarté.** Une réponse de substitution rédigée à l'avance : elle serait fausse pour
toute question autre que celle prévue.

**Coût.** Les extraits ne reformulent pas. Une question posée avec d'autres mots que
ceux des documents est mal servie : c'est la limite visible dans l'évaluation.

## D4. Recherche hybride, modèle d'embeddings local

**Choix.** BM25 et vecteurs fusionnés par rangs réciproques. Embeddings calculés
localement par un modèle multilingue de 220 Mo.

**Pourquoi.** BM25 retrouve les termes exacts (« P1 », « C-12 », « 35 % ») ; les vecteurs
rapprochent les paraphrases. Un modèle local ne coûte rien par requête et n'envoie
aucun document à un tiers.

**Écarté.** Embeddings par API (coût, dépendance, données sortantes) ; vecteurs seuls
(mauvais sur les codes et les montants).

**Coût.** Le modèle lit environ 450 caractères par passage : la fin des passages les
plus longs n'est vue que par BM25. Sur le jeu d'évaluation, l'apport des vecteurs se
voit sur le rang du bon document (MRR 0,93 contre 0,88), pas sur les réponses en mode
extraits.

## D5. Deux stockages de vecteurs derrière une interface

**Choix.** Fichier NumPy par défaut, serveur Qdrant si `QDRANT_URL` est renseigné.

**Pourquoi.** À l'échelle d'un prototype (118 passages), un fichier suffit et supprime
toute infrastructure. L'interface permet de passer à un serveur sans toucher au reste.

**Coût.** Le chemin Qdrant n'est testé qu'avec le client embarqué en mémoire, jamais
contre un serveur.

## D6. Un agent borné, qui propose

**Choix.** Une boucle à deux nœuds, quatre outils, un nombre de tours limité. L'agent
ne crée rien : il enregistre une proposition qu'une personne approuve ou refuse.

**Pourquoi.** L'action qui engage l'entreprise (ouvrir une intervention) reste une
décision humaine, tracée avec le nom du valideur. Les outils n'acceptent pas
d'identifiant client, ce qui retire au modèle le moyen de sortir du dossier.

**Écarté.** Plusieurs agents qui se coordonnent : coût et comportement difficiles à
prévoir pour un besoin qui tient en quatre outils. Une pause de graphe en mémoire
(`interrupt`) pour la validation : la proposition serait perdue au redémarrage, alors
qu'une ligne en base survit et s'audite.

**Coût.** Une étape humaine de plus pour chaque intervention.

## D7. Dans le doute, transmettre

**Choix.** Confiance de recherche sous le seuil, réponse sans citation, passage trop
peu lié à la question, erreur interne : la demande va à un gestionnaire.

**Pourquoi.** Une réponse fausse sur un délai ou un tarif coûte plus cher qu'une
réponse tardive.

**Coût, mesuré.** La règle « au moins deux termes partagés » a fait passer les questions
hors sujet répondues à tort de 3 sur 13 à 0 sur 17 dans les essais, et fait transmettre
trois questions du jeu de développement auxquelles le système répondait avant (dont deux
réponses étaient fausses). Détail dans [05-evaluation.md](05-evaluation.md).

## D8. Plusieurs fournisseurs de modèles, un modèle par tâche

**Choix.** LiteLLM comme point d'entrée unique ; le modèle de chaque tâche est un
paramètre.

**Pourquoi.** La rédaction d'une réponse courte et la conduite d'un dossier avec outils
n'ont pas les mêmes exigences ni le même prix. Changer de fournisseur ne doit pas
demander de réécriture.

**Coût.** Une dépendance de plus, et un comportement qui varie selon les fournisseurs
(les paramètres d'échantillonnage ne sont jamais envoyés, certains modèles récents les
refusent).

## D9. SQLite par défaut, PostgreSQL par configuration

**Choix.** SQLAlchemy, types communs aux deux bases.

**Pourquoi.** Démarrer sans serveur de base de données ; passer à PostgreSQL par une
variable d'environnement.

**Coût.** PostgreSQL n'a jamais été exécuté ; pas de migrations.

## Ce qui n'a pas été fait, volontairement

- Entraînement ou ajustement de modèle : rien dans le besoin ne le demande.
- Interface en framework web complet : Streamlit suffit pour valider le parcours.
- Détection des données personnelles par modèle (noms, adresses) : expressions
  régulières avec contrôle des clés, limite documentée dans
  [06-securite-rgpd.md](06-securite-rgpd.md).
