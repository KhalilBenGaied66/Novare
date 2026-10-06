# Rapport d'évaluation

- Date : 2026-10-06T17:19:29+02:00
- Recherche : hybrid (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- Index : 118 passages, top-k 4, seuil de confiance 0.35
- Modèles : rag = ollama_chat/qwen3.5:4b, agent = ollama_chat/qwen3.5:9b, judge = ollama_chat/qwen3.5:9b

| Mesure | Tous | dev | test |
|---|---|---|---|
| Cas | 136 | 85 | 51 |
| Routage (triage) | 0.9853 | 1.0 | 0.9608 |
| Route finale | 0.9191 | 0.9059 | 0.9412 |
| Document attendu retrouvé (top-k) | 1.0 | 1.0 | 1.0 |
| MRR | 0.8889 | 0.9204 | 0.8299 |
| Document attendu cité (demandes à répondre) | 1.0 | 1.0 | 1.0 |
| Faits attendus dans la réponse | 0.8857 | 0.8864 | 0.8846 |
| Données personnelles détectées | 1.0 | 1.0 | 1.0 |
| Documents interdits atteints | 0 | 0 | 0 |
| Faits interdits dans la réponse | 0 | 0 | 0 |
| Fidélité (juge LLM) | 0.9857 | 1.0 | 0.9545 |
| Latence moyenne (ms, indicatif) | 3032 | 2751 | 3501 |
| Latence p95 (ms, indicatif) | 18633 | 16488 | 20503 |
| Coût total (€) | 0.0 | 0.0 | 0.0 |

## Seuils
Tous les seuils de `evals/thresholds.json` sont respectés.

## Cas en échec (19)
- G-024 (dev, agent) : faits absents de la réponse : ['8 h ouvrees', 'priorite p2']
- G-025 (dev, agent) : faits absents de la réponse : ['24 h/24']
- G-034 (test, agent) : faits absents de la réponse : ['client non identifie']
- G-037 (dev, rag) : faits absents de la réponse : ['8 h ouvrees']
- G-055 (dev, rag) : faits absents de la réponse : ['sans tacite reconduction']
- G-069 (test, rag) : faits absents de la réponse : ['astreinte incluse 24 h/24']
- G-078 (dev, rag) : faits absents de la réponse : ['211,95']
- G-081 (test, rag) : faits absents de la réponse : ['rappelle le client dans les 20 minutes|rappel du client.*20 minutes']
- G-104 (dev, out_of_scope) : route finale rag, attendue human
- G-113 (dev, rag) : route finale rag, attendue human
- G-114 (dev, rag) : route finale rag, attendue human
- G-122 (dev, out_of_scope) : route finale rag, attendue human
- G-123 (dev, out_of_scope) : route finale rag, attendue human
- G-124 (dev, out_of_scope) : route finale rag, attendue human
- G-125 (dev, out_of_scope) : route finale rag, attendue human
- G-126 (dev, out_of_scope) : route finale rag, attendue human
- G-131 (test, human) : route rag, attendue human ; route finale rag, attendue human
- G-132 (test, agent) : route rag, attendue agent ; route finale rag, attendue agent
- G-133 (test, rag) : route finale rag, attendue human ; réponse jugée non fidèle aux sources
