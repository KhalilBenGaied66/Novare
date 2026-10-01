"""Export the requests rated KO by handlers as candidate cases for the golden set.

A KO feedback says the pipeline got a request wrong. Each one becomes a draft case in
`evals/candidates.json`, with the masked question and what the pipeline did; a person
then writes the expected route, documents and facts before adding it to
`evals/golden_set.json`. Nothing is added to the golden set automatically.

    PYTHONPATH=backend python scripts/feedback_to_golden.py
"""

import json
import sys

from sqlalchemy import select

from app.core.config import get_settings
from app.db import session
from app.db.models import FeedbackRow, RequestLogRow


def candidates() -> list[dict]:
    statement = (
        select(FeedbackRow, RequestLogRow)
        .join(RequestLogRow, RequestLogRow.request_id == FeedbackRow.request_id)
        .where(FeedbackRow.ok.is_(False))
        .order_by(FeedbackRow.ts)
    )
    with session.session_scope() as db:
        return [
            {
                "request_id": request.request_id,
                "input": {"q": request.q_masked, "client_id": request.client_id, "montant": None},
                "observed_route": request.route,
                "observed_mode": request.mode,
                "handler_comment": feedback.comment,
                "expected_route": None,
                "expected_final_route": None,
                "expected_docs": [],
                "expected_facts": [],
                "note": "À compléter par un gestionnaire avant ajout au jeu de référence.",
            }
            for feedback, request in db.execute(statement)
        ]


def main() -> int:
    session.init_db()
    found = candidates()
    path = get_settings().evals_dir / "candidates.json"
    path.write_text(json.dumps(found, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(found)} cas candidat(s) écrit(s) dans {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
