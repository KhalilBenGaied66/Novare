"""Check the LLM path for real: one documentary question and one dossier, with a provider key.

The test suite replaces the provider with a stub, so it cannot tell whether the configured
models answer, call the tools and cite their sources. Run this once on a machine that
has the keys of `RAG_MODEL` and `AGENT_MODEL` (see `.env.example`):

    PYTHONPATH=backend python scripts/smoke_llm.py

It uses the configured corpus and a temporary database, prints what each request cost,
and exits with 1 if a request was not answered by a model.
"""

import sys

from app.core import llm
from app.core.schemas import AskRequest, AskResponse
from app.db import session
from app.ingestion import ingest
from app.retrieval import hybrid
from app.services import ask

REQUESTS = [
    ("rag", AskRequest(q="Quelle est la majoration appliquée pour un déplacement le week-end ?")),
    (
        "agent",
        AskRequest(
            q="La chaudière de la chaufferie est en panne depuis ce matin, merci d'intervenir.",
            client_id="C-12",
        ),
    ),
]


def main() -> int:
    missing = [task for task, _ in REQUESTS if not llm.is_enabled(task)]
    if missing:
        models = ", ".join(f"{task} = {llm.model_for(task)}" for task in missing)
        print(f"Aucune clé de fournisseur pour : {models}. Renseignez-la dans .env.")
        return 1

    failures = 0
    with session.temporary_database():
        if not hybrid.index_status()["ready"]:
            ingest.ingest_docs()
        for task, request in REQUESTS:
            response = ask.handle_ask(request)
            _show(task, response)
            if response.mode != "llm":
                failures += 1
                print("  ÉCHEC : la réponse ne vient pas d'un modèle.")
                for line in response.decision_log:
                    print(f"    {line}")
    return 1 if failures else 0


def _show(task: str, response: AskResponse) -> None:
    usage = response.usage
    print()
    print(f"[{task}] route={response.route} mode={response.mode}")
    print(
        f"  modèle={usage.model} appels={usage.llm_calls} "
        f"tokens={usage.prompt_tokens}+{usage.completion_tokens} coût={usage.cost_eur:.6f} €"
    )
    print(f"  sources citées : {[citation.doc for citation in response.citations]}")
    for line in response.answer.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
