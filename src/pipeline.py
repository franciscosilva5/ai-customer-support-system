from src.classifier import classify_ticket
from src.escalation import decide_escalation
from src.generation import generate_support_response
from src.retrieval import KnowledgeRetriever


class CustomerSupportPipeline:
    def __init__(self):
        self.retriever = KnowledgeRetriever()

    def process(self, ticket):
        classification = classify_ticket(
            ticket
        )

        retrieval_results = self.retriever.search(
            ticket,
            top_k=3,
        )

        escalation = decide_escalation(
            classification,
            retrieval_results,
        )

        if escalation["escalate"]:
            response = (
                "I don't have enough reliable information "
                "to resolve this request automatically. "
                "This ticket should be reviewed by "
                "NovaDesk Support."
            )
        else:
            response = generate_support_response(
                ticket,
                retrieval_results,
            )

        return {
            "ticket": ticket,
            "classification": classification,
            "retrieval_results": retrieval_results,
            "escalation": escalation,
            "response": response,
        }
