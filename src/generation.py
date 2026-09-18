from openai import OpenAI

from src.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


SYSTEM_PROMPT = """
You are NovaDesk's AI customer support assistant.

Your job is to answer the customer's question using ONLY the
knowledge-base context provided to you.

Rules:
- Do not invent policies, prices, timelines, or procedures.
- If the context does not contain enough information, say that the
  issue should be reviewed by NovaDesk Support.
- Be concise, helpful, and professional.
- Do not claim that an action has already been performed.
- Do not promise a refund, account recovery, or billing adjustment.
- When appropriate, give clear next steps.
"""


def build_context(retrieval_results):
    sections = []

    for index, result in enumerate(
        retrieval_results,
        start=1,
    ):
        sections.append(
            f"""SOURCE {index}
Title: {result["title"]}
File: {result["source"]}

{result["text"]}
"""
        )

    return "\n\n".join(sections)


def generate_support_response(
    ticket,
    retrieval_results,
):
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=OPENAI_API_KEY
    )

    context = build_context(
        retrieval_results
    )

    user_prompt = f"""
CUSTOMER TICKET:
{ticket}

KNOWLEDGE BASE:
{context}

Write a grounded support response to the customer.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
    )

    return response.output_text.strip()
