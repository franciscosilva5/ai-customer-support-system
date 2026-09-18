from src.classifier import classify_ticket
from src.escalation import decide_escalation
from src.retrieval import KnowledgeRetriever


def test_billing_classification():
    result = classify_ticket(
        "I was charged twice for my subscription."
    )

    assert result["category"] == "billing"
    assert result["confidence"] >= 0.70


def test_unknown_classification():
    result = classify_ticket(
        "My laptop screen is broken."
    )

    assert result["category"] == "unknown"
    assert result["confidence"] == 0.0


def test_retrieval_duplicate_charge():
    retriever = KnowledgeRetriever()

    results = retriever.search(
        "I was charged twice for my subscription.",
        top_k=3,
    )

    assert results[0]["title"] == "Duplicate Charges"


def test_supported_ticket_not_escalated():
    retriever = KnowledgeRetriever()

    ticket = "I forgot my password and cannot log in."

    classification = classify_ticket(ticket)
    results = retriever.search(ticket, top_k=3)

    decision = decide_escalation(
        classification,
        results,
    )

    assert decision["escalate"] is False
    assert decision["reasons"] == []


def test_unknown_ticket_is_escalated():
    retriever = KnowledgeRetriever()

    ticket = "The app is behaving strangely today."

    classification = classify_ticket(ticket)
    results = retriever.search(ticket, top_k=3)

    decision = decide_escalation(
        classification,
        results,
    )

    assert decision["escalate"] is True
    assert len(decision["reasons"]) > 0
