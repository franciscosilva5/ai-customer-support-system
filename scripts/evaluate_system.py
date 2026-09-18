import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.classifier import classify_ticket
from src.escalation import decide_escalation
from src.retrieval import KnowledgeRetriever


TEST_CASES = [
    {
        "ticket": "I was charged twice for my subscription.",
        "expected_category": "billing",
        "expected_sources": ["Duplicate Charges"],
        "expected_escalation": False,
    },
    {
        "ticket": "I need a refund for a duplicate payment.",
        "expected_category": "billing",
        "expected_sources": [
            "Duplicate Charges",
            "Refund Policy",
        ],
        "expected_escalation": False,
    },
    {
        "ticket": "I forgot my password and cannot log in.",
        "expected_category": "account_access",
        "expected_sources": ["Password and Login Problems"],
        "expected_escalation": False,
    },
    {
        "ticket": "I lost access to my authenticator app.",
        "expected_category": "security",
        "expected_sources": ["Two-Factor Authentication"],
        "expected_escalation": False,
    },
    {
        "ticket": "How do I delete my workspace?",
        "expected_category": "account_management",
        "expected_sources": ["Account and Workspace Deletion"],
        "expected_escalation": False,
    },
    {
        "ticket": "How can I export my workspace data?",
        "expected_category": "data",
        "expected_sources": ["Exporting Workspace Data"],
        "expected_escalation": False,
    },
    {
        "ticket": "How do I cancel my subscription?",
        "expected_category": "billing",
        "expected_sources": ["Canceling a Subscription"],
        "expected_escalation": False,
    },
    {
        "ticket": "The app is behaving strangely today.",
        "expected_category": "unknown",
        "expected_sources": None,
        "expected_escalation": True,
    },
    {
        "ticket": "Can NovaDesk order lunch for my team?",
        "expected_category": "unknown",
        "expected_sources": None,
        "expected_escalation": True,
    },
    {
        "ticket": "My laptop screen is broken.",
        "expected_category": "unknown",
        "expected_sources": None,
        "expected_escalation": True,
    },
]


def main():
    retriever = KnowledgeRetriever()

    category_correct = 0
    hit_at_1_correct = 0
    hit_at_3_correct = 0
    escalation_correct = 0
    supported_cases = 0

    print("\nAI CUSTOMER SUPPORT SYSTEM EVALUATION")
    print("=" * 60)

    for index, case in enumerate(TEST_CASES, start=1):
        ticket = case["ticket"]

        classification = classify_ticket(ticket)

        results = retriever.search(
            ticket,
            top_k=3,
        )

        escalation = decide_escalation(
            classification,
            results,
        )

        predicted_category = classification["category"]
        predicted_escalation = escalation["escalate"]

        top_titles = [
            result["title"]
            for result in results
        ]

        category_ok = (
            predicted_category
            == case["expected_category"]
        )

        escalation_ok = (
            predicted_escalation
            == case["expected_escalation"]
        )

        if category_ok:
            category_correct += 1

        if escalation_ok:
            escalation_correct += 1

        if case["expected_sources"] is not None:
            supported_cases += 1

            hit_at_1 = (
                top_titles[0]
                in case["expected_sources"]
            )

            hit_at_3 = any(
                title in case["expected_sources"]
                for title in top_titles
            )

            if hit_at_1:
                hit_at_1_correct += 1

            if hit_at_3:
                hit_at_3_correct += 1

        else:
            hit_at_1 = None
            hit_at_3 = None

        print(f"\nCASE {index}")
        print("Ticket:", ticket)

        print(
            "Category:",
            predicted_category,
            "| expected:",
            case["expected_category"],
            "|",
            "PASS" if category_ok else "FAIL",
        )

        if case["expected_sources"] is not None:
            print(
                "Top source:",
                top_titles[0],
                "| expected:",
                case["expected_sources"],
                "|",
                "PASS" if hit_at_1 else "FAIL",
            )

            print(
                "Hit@3:",
                top_titles,
                "|",
                "PASS" if hit_at_3 else "FAIL",
            )
        else:
            print(
                "Top source:",
                top_titles[0],
                "| expected: out-of-scope",
            )

        print(
            "Escalation:",
            predicted_escalation,
            "| expected:",
            case["expected_escalation"],
            "|",
            "PASS" if escalation_ok else "FAIL",
        )

    total = len(TEST_CASES)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Category accuracy: "
        f"{category_correct}/{total} "
        f"({category_correct / total:.1%})"
    )

    print(
        f"Retrieval Hit@1: "
        f"{hit_at_1_correct}/{supported_cases} "
        f"({hit_at_1_correct / supported_cases:.1%})"
    )

    print(
        f"Retrieval Hit@3: "
        f"{hit_at_3_correct}/{supported_cases} "
        f"({hit_at_3_correct / supported_cases:.1%})"
    )

    print(
        f"Escalation accuracy: "
        f"{escalation_correct}/{total} "
        f"({escalation_correct / total:.1%})"
    )


if __name__ == "__main__":
    main()
