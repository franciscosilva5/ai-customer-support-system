import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import CustomerSupportPipeline


st.set_page_config(
    page_title="AI Customer Support System",
    page_icon="🎧",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    return CustomerSupportPipeline()


pipeline = load_pipeline()


st.title("🎧 AI Customer Support System")

st.write(
    "Enter a customer support ticket and the system will classify it, "
    "retrieve relevant knowledge-base articles, decide whether human "
    "escalation is needed, and generate a grounded response."
)


with st.sidebar:
    st.header("System")

    st.markdown(
        """
        **Pipeline**
        - Ticket classification
        - Semantic retrieval
        - Cross-encoder reranking
        - Escalation decision
        - Grounded LLM response
        """
    )

    st.header("Evaluation")

    st.markdown(
        """
        **Small domain-specific evaluation**
        - Category accuracy: 100%
        - Retrieval Hit@1: 100%
        - Retrieval Hit@3: 100%
        - Escalation accuracy: 100%
        """
    )

    st.caption(
        "Evaluation uses a small curated test set and is not a general benchmark."
    )

    st.header("Demo Context")

    st.caption(
        "NovaDesk is a fictional SaaS company. "
        "The knowledge base is synthetic and created for this portfolio project."
    )


ticket = st.text_area(
    "Customer ticket",
    height=140,
    placeholder="Example: I was charged twice for my subscription.",
)


if st.button(
    "Process ticket",
    type="primary",
    use_container_width=True,
):
    if not ticket.strip():
        st.warning("Enter a customer ticket first.")
        st.stop()

    with st.spinner("Processing ticket..."):
        try:
            result = pipeline.process(ticket.strip())
        except Exception as error:
            st.error(
                f"An error occurred while processing the ticket: {error}"
            )
            st.stop()

    classification = result["classification"]
    retrieval_results = result["retrieval_results"]
    escalation = result["escalation"]

    st.divider()
    st.subheader("Support Response")

    if escalation["escalate"]:
        st.warning("Human escalation recommended")
    else:
        st.success("Automatic response available")

    st.write(result["response"])

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Category",
            classification["category"]
            .replace("_", " ")
            .title(),
        )

    with col2:
        st.metric(
            "Classifier confidence",
            f'{classification["confidence"] * 100:.0f}%',
        )

    with col3:
        st.metric(
            "Escalation",
            "Yes" if escalation["escalate"] else "No",
        )

    if escalation["reasons"]:
        st.subheader("Escalation Reasons")

        for reason in escalation["reasons"]:
            st.write(f"- {reason}")

    st.divider()
    st.subheader("Knowledge Sources")

    top_result = retrieval_results[0]

    weak_evidence = (
        top_result["retrieval_score"] < 0.25
        or top_result["rerank_score"] < -3.0
    )

    if escalation["escalate"] and weak_evidence:
        st.info(
            "The retrieved articles below are low-relevance candidates. "
            "They are shown for transparency and were not considered "
            "reliable enough for automatic resolution."
        )

    for rank, source in enumerate(
        retrieval_results,
        start=1,
    ):
        label = f'{rank}. {source["title"]}'

        if (
            source["retrieval_score"] < 0.25
            or source["rerank_score"] < -3.0
        ):
            label += " — low relevance"

        with st.expander(label):
            st.write(
                f'**Semantic score:** '
                f'{source["retrieval_score"]:.3f}'
            )

            st.write(
                f'**Reranker score:** '
                f'{source["rerank_score"]:.3f}'
            )

            st.write(source["text"])

    st.divider()
    st.subheader("Pipeline")

    st.code(
        "Customer Ticket\n"
        "      ↓\n"
        "Classification\n"
        "      ↓\n"
        "Semantic Retrieval\n"
        "      ↓\n"
        "Cross-Encoder Reranking\n"
        "      ↓\n"
        "Escalation Decision\n"
        "      ↓\n"
        "Grounded Response",
        language=None,
    )
