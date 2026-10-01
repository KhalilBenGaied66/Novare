# Rapport d'évaluation

- Date : 2026-10-02T00:52:06+02:00
- Recherche : hybrid (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Index : 118 passages, top-k 4, seuil de confiance 0.35
- Modèles : aucun LLM (réponses extractives et plan d'agent déterministe)

| Mesure | Tous | dev | test |
|---|---|---|---|
| Cas | 136 | 85 | 51 |
| Routage (triage) | 0.9853 | 1.0 | 0.9608 |
| Route finale | 0.9044 | 0.8824 | 0.9412 |
| Document attendu retrouvé (top-k) | 1.0 | 1.0 | 1.0 |
| MRR | 0.8889 | 0.9204 | 0.8299 |
| Document attendu cité (demandes à répondre) | 0.8986 | 0.8667 | 0.9583 |
| Faits attendus dans la réponse | 0.8714 | 0.8636 | 0.8846 |
| Données personnelles détectées | 1.0 | 1.0 | 1.0 |
| Documents interdits atteints | 0 | 0 | 0 |
| Faits interdits dans la réponse | 0 | 0 | 0 |
| Fidélité (juge LLM) | — | — | — |
| Latence moyenne (ms, indicatif) | 10 | 10 | 10 |
| Latence p95 (ms, indicatif) | 25 | 24 | 25 |
| Coût total (€) | 0.0 | 0.0 | 0.0 |

## Seuils
Tous les seuils de `evals/thresholds.json` sont respectés.

## Cas en échec (20)
- G-078 (dev, rag) : route finale human, attendue rag ; aucun document attendu cité (rien) ; faits absents de la réponse : ['211,95']
- G-079 (test, rag) : aucun document attendu cité (politique_donnees_personnelles.md) ; faits absents de la réponse : ['devis prealable']
- G-080 (dev, rag) : aucun document attendu cité (faq_gestionnaires.md) ; faits absents de la réponse : ['30 jours']
- G-081 (test, rag) : faits absents de la réponse : ['rappelle le client dans les 20 minutes|rappel du client.*20 minutes']
- G-082 (dev, rag) : route finale human, attendue rag ; aucun document attendu cité (rien) ; faits absents de la réponse : ['24 heures ouvrees']
- G-114 (dev, rag) : route finale rag, attendue human
- G-120 (dev, rag) : route finale human, attendue rag ; aucun document attendu cité (rien)
- G-122 (dev, out_of_scope) : route finale rag, attendue human
- G-123 (dev, out_of_scope) : route finale rag, attendue human
- G-124 (dev, out_of_scope) : route finale rag, attendue human
- G-125 (dev, out_of_scope) : route finale rag, attendue human
- G-126 (dev, out_of_scope) : route finale rag, attendue human
- G-127 (dev, out_of_scope) : route finale rag, attendue human
- G-128 (dev, rag) : aucun document attendu cité (procedure_litiges_facturation.md) ; faits absents de la réponse : ['45 jours']
- G-129 (dev, rag) : aucun document attendu cité (conditions_generales_maintenance.md) ; faits absents de la réponse : ['deux mois']
- G-130 (dev, rag) : faits absents de la réponse : ['ne se cumulent pas']
- G-131 (test, human) : route rag, attendue human ; route finale rag, attendue human
- G-132 (test, agent) : route rag, attendue agent ; route finale human, attendue agent
- G-133 (test, rag) : route finale rag, attendue human
- G-134 (test, agent) : faits absents de la réponse : ['priorite p1']
