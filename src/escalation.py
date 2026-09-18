CLASSIFIER_CONFIDENCE_THRESHOLD = 0.70
RETRIEVAL_SCORE_THRESHOLD = 0.25
RERANK_SCORE_THRESHOLD = -3.0


def decide_escalation(
    classification,
    retrieval_results,
):
    reasons = []

    category = classification["category"]
    classifier_confidence = classification["confidence"]

    if category == "unknown":
        reasons.append("unknown ticket category")

    if classifier_confidence < CLASSIFIER_CONFIDENCE_THRESHOLD:
        reasons.append("low classifier confidence")

    if not retrieval_results:
        reasons.append("no knowledge-base evidence")

        return {
            "escalate": True,
            "reasons": reasons,
        }

    top_result = retrieval_results[0]

    retrieval_score = top_result.get(
        "retrieval_score",
        0.0,
    )

    rerank_score = top_result.get(
        "rerank_score",
        float("-inf"),
    )

    if retrieval_score < RETRIEVAL_SCORE_THRESHOLD:
        reasons.append("weak semantic retrieval")

    if rerank_score < RERANK_SCORE_THRESHOLD:
        reasons.append("weak reranker evidence")

    return {
        "escalate": len(reasons) > 0,
        "reasons": reasons,
    }
