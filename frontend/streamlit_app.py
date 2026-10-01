"""Streamlit interface of the DossierOps assistant.

The page collects a request, sends it to the API through `api_client` and displays the
response. Routing and business rules belong to the API: nothing here decides how a
request is handled. The only processing done on this side is turning the typed amount
into a number, and refusing to send what cannot be read as one.

Run with: streamlit run frontend/streamlit_app.py
"""

import re
from dataclasses import dataclass

import api_client
import streamlit as st

# Session-state keys: what the page remembers between two runs of the script.
KEY_FORM_VERSION = "form_version"  # number of examples loaded so far
KEY_PREFILL = "prefill"  # Example whose values initialise the request form
KEY_RESULT = "result"  # last AskResponse, as a dict
KEY_SENT = "sent"  # client and amount actually sent with that request
KEY_ACTION_OUTCOME = "action_outcome"  # ActionResult once the proposal is decided
KEY_FEEDBACK_SENT = "feedback_sent"
PRIORITIES = ("P1", "P2", "P3")
KEY_VALIDATOR_NAME = "validator_name"  # last validator name, to avoid retyping it

ROUTE_LABELS = {
    "automation": "Automatisation par règle métier",
    "rag": "Réponse documentaire",
    "agent": "Agent de traitement de dossier",
    "human": "Transmission à un gestionnaire",
}
ROUTE_COLORS = {"automation": "green", "rag": "blue", "agent": "violet", "human": "orange"}
MODE_LABELS = {
    "rule": "règle métier, sans LLM",
    "llm": "rédaction par un LLM",
    "extractive": "extraits des documents, sans LLM",
    "deterministic": "plan déterministe, sans LLM",
    "none": "aucune réponse automatique",
}
RETRIEVAL_LABELS = {"hybrid": "hybride (lexicale et vectorielle)", "bm25": "lexicale (BM25)"}
MISSING = "—"

_AMOUNT = re.compile(r"[0-9]+(?:[.,][0-9]{1,2})?")


@dataclass(frozen=True)
class Example:
    """A sample request. Fields hold what would be typed in the form; "" = left empty."""

    label: str
    q: str
    client_id: str
    montant: str


# Initial content of the form: nothing is prefilled, neither a client nor an amount.
BLANK = Example(label="", q="", client_id="", montant="")

EXAMPLES = (
    Example(
        label="Litige de 120 €",
        q="Je conteste ma facture du mois de mars : le déplacement a été facturé deux fois.",
        client_id="C-12",
        montant="120",
    ),
    Example(
        label="Tarif week-end",
        q="Quelle est la majoration appliquée pour un déplacement le week-end ?",
        client_id="",
        montant="",
    ),
    Example(
        label="Panne de chaudière",
        q=(
            "La chaudière de l'immeuble est en panne depuis ce matin. "
            "Pouvez-vous planifier une intervention ?"
        ),
        client_id="C-12",
        montant="",
    ),
)


def parse_amount(text: str) -> float | None:
    """Convert the amount field to a number; an empty field means "not provided".

    Accepts `120`, `120,50`, `120.50`, `1 250,50` and a trailing `€`. Anything else,
    zero included, raises `ValueError`: the form never turns an empty or unreadable
    value into a number, because an amount changes how the request is routed.
    """
    compact = "".join(text.split())
    if not compact:
        return None
    compact = compact.removesuffix("€")
    if not _AMOUNT.fullmatch(compact):
        raise ValueError("unreadable amount")
    amount = float(compact.replace(",", "."))
    if amount <= 0:
        raise ValueError("amount must be positive")
    return amount


def format_number(value: float, decimals: int) -> str:
    """French number formatting: decimal comma."""
    return f"{value:.{decimals}f}".replace(".", ",")


def format_percent(share: float | None) -> str:
    """A 0..1 share as a percentage; the API sends None when there is nothing to measure."""
    return MISSING if share is None else f"{round(share * 100)} %"


def format_ms(value: int | None) -> str:
    return MISSING if value is None else f"{value} ms"


def usage_summary(usage: dict) -> str:
    if usage["llm_calls"] == 0:
        return "Aucun LLM utilisé pour cette demande : coût nul."
    return (
        f"Modèle {usage['model'] or 'non précisé'} : {usage['llm_calls']} appel(s), "
        f"{usage['prompt_tokens']} jetons en entrée, {usage['completion_tokens']} en sortie, "
        f"coût {format_number(usage['cost_eur'], 4)} €."
    )


def sent_summary(sent: dict) -> str:
    """What was sent with the request, so that an omitted field is visible as omitted."""
    client = sent["client_id"] or "non renseigné"
    montant = sent["montant"]
    amount = "non renseigné" if montant is None else f"{format_number(montant, 2)} €"
    return f"Demande envoyée avec : client {client} ; montant du litige {amount}."


def clear_result() -> None:
    """Forget the displayed response and everything attached to it."""
    for key in (KEY_RESULT, KEY_SENT, KEY_ACTION_OUTCOME, KEY_FEEDBACK_SENT):
        st.session_state.pop(key, None)


def load_example(example: Example) -> None:
    """Button callback: show the example in a brand-new request form.

    The version is part of the form id and of the widget keys, so the browser builds
    new fields initialised with the example. Assigning the values to the existing
    fields is not enough: for a field edited but not yet submitted, the browser shows
    the assigned value and still sends the typed one on submit (observed with
    Streamlit 1.64), i.e. the request would not match what the form displays.
    """
    st.session_state[KEY_PREFILL] = example
    st.session_state[KEY_FORM_VERSION] = st.session_state.get(KEY_FORM_VERSION, 0) + 1
    clear_result()


def submit_request(q: str, client_text: str, amount_text: str) -> None:
    """Send the form to the API and keep the response for display."""
    clear_result()
    q = q.strip()
    if not q:
        st.warning("Saisissez la demande du client avant de lancer le traitement.")
        return
    try:
        montant = parse_amount(amount_text)
    except ValueError:
        st.error(
            "Montant invalide. Saisissez un nombre positif, par exemple 120 ou 120,50, "
            "ou laissez le champ vide."
        )
        return
    # An empty field is sent as None (JSON null): no default client, no 0.0 amount.
    client_id = client_text.strip() or None
    try:
        with st.spinner("Traitement de la demande en cours..."):
            result = api_client.ask(q=q, client_id=client_id, montant=montant)
    except api_client.ApiError as error:
        st.error(str(error))
        return
    st.session_state[KEY_RESULT] = result
    st.session_state[KEY_SENT] = {"client_id": client_id, "montant": montant}


def decide_action(
    action_id: str, approved: bool, validator: str, reason: str, priority: str
) -> None:
    validator = validator.strip()
    if not validator:
        st.warning("Indiquez le nom du valideur avant de valider ou de refuser.")
        return
    try:
        if approved:
            outcome = api_client.approve(
                action_id=action_id, validator=validator, priority=priority
            )
        else:
            outcome = api_client.reject(
                action_id=action_id, validator=validator, reason=reason.strip()
            )
    except api_client.ApiError as error:
        st.error(str(error))
        return
    st.session_state[KEY_ACTION_OUTCOME] = outcome
    st.session_state[KEY_VALIDATOR_NAME] = validator
    # Rerun so that the outcome replaces the form instead of appearing under it.
    st.rerun()


def send_feedback(request_id: str, ok: bool, comment: str) -> None:
    try:
        api_client.feedback(request_id=request_id, ok=ok, comment=comment.strip())
    except api_client.ApiError as error:
        st.error(str(error))
        return
    st.session_state[KEY_FEEDBACK_SENT] = True
    st.rerun()


def render_examples() -> None:
    st.caption("Exemples de demandes (remplissent le formulaire, sans l'envoyer)")
    columns = st.columns(len(EXAMPLES))
    for index, (column, example) in enumerate(zip(columns, EXAMPLES, strict=True)):
        column.button(
            example.label,
            key=f"example_{index}",
            on_click=load_example,
            args=(example,),
            width="stretch",
        )


def render_request_form() -> None:
    version = st.session_state.get(KEY_FORM_VERSION, 0)
    prefill = st.session_state.get(KEY_PREFILL, BLANK)
    with st.form(f"request_form_{version}"):
        q = st.text_area("Demande du client", value=prefill.q, key=f"q_{version}", height=140)
        left, right = st.columns(2)
        client_text = left.text_input(
            "Identifiant client (facultatif)",
            value=prefill.client_id,
            key=f"client_id_{version}",
            help="Format : C-12. Laissez vide si le client n'est pas identifié.",
        )
        amount_text = right.text_input(
            "Montant du litige en € (facultatif)",
            value=prefill.montant,
            key=f"montant_{version}",
            help="Exemples : 120 ou 120,50. Laissez vide s'il n'y a pas de montant contesté.",
        )
        submitted = st.form_submit_button("Traiter la demande", type="primary", key="submit_ask")
    if submitted:
        submit_request(q, client_text, amount_text)


def render_result(result: dict, sent: dict) -> None:
    st.subheader("Résultat")
    route = result["route"]
    mode = result["mode"]
    with st.container(horizontal=True):
        st.badge(f"Route : {ROUTE_LABELS.get(route, route)}", color=ROUTE_COLORS.get(route, "gray"))
        st.badge(f"Mode : {MODE_LABELS.get(mode, mode)}", color="gray")
    st.caption(sent_summary(sent))

    with st.container(border=True):
        # Two trailing spaces force a Markdown line break: the answer keeps its lines.
        st.markdown(result["answer"].replace("\n", "  \n"))
    if result["ticket_id"]:
        st.success(f"Ticket {result['ticket_id']} créé automatiquement.")
    if result["needs_validation"]:
        st.caption("Réponse à relire par un gestionnaire avant tout envoi au client.")

    left, middle, right = st.columns(3)
    left.metric("Confiance", format_percent(result["confidence"]))
    middle.metric("Latence", format_ms(result["latency_ms"]))
    right.metric("Coût LLM", f"{format_number(result['usage']['cost_eur'], 4)} €")
    st.caption(usage_summary(result["usage"]))
    if result["retrieval_mode"] != "none":
        retrieval = result["retrieval_mode"]
        st.caption(f"Recherche documentaire : {RETRIEVAL_LABELS.get(retrieval, retrieval)}.")
    if result["pii_redacted"]:
        st.caption(
            "Données personnelles masquées avant traitement : "
            + ", ".join(result["pii_redacted"])
            + "."
        )

    render_citations(result["citations"])
    with st.expander(f"Journal de décision ({len(result['decision_log'])} étapes)"):
        lines = [f"{number}. {step}" for number, step in enumerate(result["decision_log"], 1)]
        # st.text shows the lines as they are: no Markdown interpretation of "[1]" or "*".
        st.text("\n".join(lines))
    st.caption(f"Identifiant de la demande : {result['request_id']}")


def render_citations(citations: list[dict]) -> None:
    if not citations:
        st.caption("Aucune source citée.")
        return
    st.markdown(f"**Sources citées ({len(citations)})**")
    for citation in citations:
        label = (
            f"[{citation['ref']}] {citation['title']} — {citation['doc']}, p. {citation['page']}"
        )
        with st.expander(label):
            if citation["section"]:
                st.caption(f"Section : {citation['section']}")
            st.text(citation["excerpt"])


def render_action(action: dict) -> None:
    st.subheader("Action proposée : création d'un ticket")
    payload = action["payload"]
    st.text(
        "\n".join(
            [
                f"Client : {payload.get('client_id') or 'non identifié'}",
                f"Priorité : {payload.get('priority') or MISSING}",
                f"Objet : {payload.get('subject') or MISSING}",
                f"Résumé : {payload.get('summary') or MISSING}",
            ]
        )
    )

    outcome = st.session_state.get(KEY_ACTION_OUTCOME)
    if outcome is not None:
        decided = outcome["action"]
        if decided["status"] == "approved":
            st.success(f"Ticket {decided['ticket_id']} créé après validation.")
        else:
            st.info("Proposition refusée : aucun ticket n'a été créé.")
        return
    if action["status"] != "pending":
        st.info("Cette proposition a déjà été traitée.")
        return

    action_id = action["action_id"]
    # Enter must not submit this form: it would trigger the first button, i.e. approve.
    with st.form("action_form", enter_to_submit=False):
        st.caption("Aucun ticket n'est créé tant que la proposition n'est pas validée.")
        validator = st.text_input(
            "Nom du valideur",
            value=st.session_state.get(KEY_VALIDATOR_NAME, ""),
            key=f"validator_{action_id}",
        )
        proposed = payload.get("priority")
        priority = st.selectbox(
            "Priorité du ticket (à confirmer ou corriger)",
            PRIORITIES,
            index=PRIORITIES.index(proposed) if proposed in PRIORITIES else 1,
            key=f"priority_{action_id}",
        )
        reason = st.text_input("Motif du refus (facultatif)", key=f"reason_{action_id}")
        left, right = st.columns(2)
        approved = left.form_submit_button(
            "Valider la création du ticket", type="primary", key="approve_action"
        )
        rejected = right.form_submit_button("Refuser", key="reject_action")
    if approved or rejected:
        decide_action(action_id, approved, validator, reason, priority)


def render_feedback(request_id: str) -> None:
    st.subheader("Votre avis sur cette réponse")
    if st.session_state.get(KEY_FEEDBACK_SENT):
        st.success("Avis enregistré. Merci.")
        return
    with st.form("feedback_form", enter_to_submit=False):
        # The key carries the request id so that a comment typed for one response is
        # never left in the field of the next one.
        comment = st.text_area("Commentaire (facultatif)", key=f"comment_{request_id}")
        left, right = st.columns(2)
        ok = left.form_submit_button("Réponse correcte (OK)", key="feedback_ok")
        ko = right.form_submit_button("Réponse à corriger (KO)", key="feedback_ko")
    if ok or ko:
        send_feedback(request_id, ok, comment)


def render_metrics(metrics: dict) -> None:
    feedback = metrics["feedback"]
    left, right = st.columns(2)
    left.metric("Demandes", metrics["requests"])
    right.metric("Escalade humaine", format_percent(metrics["escalation_rate"]))
    left.metric("Latence médiane", format_ms(metrics["latency_ms_p50"]))
    right.metric("Latence p95", format_ms(metrics["latency_ms_p95"]))
    left.metric("Demandes avec LLM", format_percent(metrics["llm_call_share"]))
    right.metric("Actions en attente", metrics["pending_actions"])
    left.metric("Avis positifs", format_percent(feedback["ok_rate"]))
    right.metric("Erreurs", metrics["errors"])
    # Kept out of the metric tiles: an amount with four decimals does not fit in one.
    st.caption(
        f"Coût LLM total : {format_number(metrics['cost_eur_total'], 4)} €. "
        f"Avis reçus : {feedback['n']}."
    )
    for route, count in metrics["by_route"].items():
        st.caption(f"{ROUTE_LABELS.get(route, route)} : {count}")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("État du service")
        try:
            readiness = api_client.ready()
        except api_client.ApiError as error:
            st.error(str(error))
            return
        if readiness["ready"]:
            st.success("API prête")
        else:
            st.warning("API non prête")
        for name, status in readiness.get("checks", {}).items():
            st.caption(f"{name} : {status}")

        st.header("Indicateurs")
        try:
            metrics = api_client.metrics()
        except api_client.ApiError as error:
            st.warning(str(error))
            return
        render_metrics(metrics)


def main() -> None:
    st.set_page_config(page_title="Novare DossierOps")
    st.title("Novare DossierOps")
    st.caption("Assistant de traitement des demandes clients (données fictives).")

    render_examples()
    render_request_form()

    result = st.session_state.get(KEY_RESULT)
    if result is not None:
        render_result(result, st.session_state[KEY_SENT])
        if result["proposed_action"] is not None:
            render_action(result["proposed_action"])
        render_feedback(result["request_id"])

    # Rendered last so that the indicators include the request handled in this run.
    render_sidebar()


if __name__ == "__main__":
    main()
