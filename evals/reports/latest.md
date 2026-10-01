# Rapport d'évaluation

- Date : 2026-10-01T14:03:57+02:00
- Recherche : hybrid (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Index : 118 passages, top-k 4, seuil de confiance 0.35
- Modèles : aucun LLM (réponses extractives et plan d'agent déterministe)

| Mesure | Tous | dev | test |
|---|---|---|---|
| Cas | 111 | 67 | 44 |
| Routage (triage) | 1.0 | 1.0 | 1.0 |
| Route finale | 0.982 | 0.9701 | 1.0 |
| Document attendu retrouvé (top-k) | 1.0 | 1.0 | 1.0 |
| MRR | 0.9272 | 0.9688 | 0.8551 |
| Document attendu cité | 0.9672 | 0.9737 | 0.9565 |
| Faits attendus dans la réponse | 0.931 | 0.9189 | 0.9524 |
| Données personnelles détectées | 1.0 | 1.0 | 1.0 |
| Documents interdits atteints | 0 | 0 | 0 |
| Faits interdits dans la réponse | 0 | 0 | 0 |
| Fidélité (juge LLM) | — | — | — |
| Latence moyenne (ms) | 38 | 39 | 35 |
| Latence p95 (ms) | 82 | 85 | 77 |
| Coût total (€) | 0.0 | 0.0 | 0.0 |

## Seuils
Tous les seuils de `evals/thresholds.json` sont respectés.

## Cas en échec (4)
- G-078 (dev, rag) : route finale human, attendue rag ; faits absents de la réponse : ['211,95']
- G-079 (test, rag) : aucun document attendu cité (politique_donnees_personnelles.md) ; faits absents de la réponse : ['devis prealable']
- G-080 (dev, rag) : aucun document attendu cité (faq_gestionnaires.md) ; faits absents de la réponse : ['30 jours']
- G-082 (dev, rag) : route finale human, attendue rag ; faits absents de la réponse : ['24 heures ouvrees']
