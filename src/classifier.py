import re


CATEGORY_PATTERNS = {
    "billing": [
        r"\bcharge\b",
        r"\bcharged\b",
        r"\bbilling\b",
        r"\binvoice\b",
        r"\bpayment\b",
        r"\brefund\b",
        r"\bsubscription\b",
        r"\brenewal\b",
        r"\bcard\b",
        r"\bpaid\b",
    ],

    "account_access": [
        r"\bpassword\b",
        r"\blog\s+in\b",
        r"\bsign\s+in\b",
        r"\blocked\s+out\b",
        r"\baccount\s+access\b",
        r"\bemail\b.*\baccess\b",
    ],

    "security": [
        r"\b2fa\b",
        r"\btwo[-\s]?factor\b",
        r"\bauthenticator\b",
        r"\brecovery\s+code\b",
        r"\bverification\s+code\b",
    ],

    "account_management": [
        r"\bdelete\b.*\baccount\b",
        r"\bdelete\b.*\bworkspace\b",
        r"\bremove\b.*\baccount\b",
        r"\bremove\b.*\bworkspace\b",
        r"\bcancel\b.*\baccount\b",
    ],

    "data": [
        r"\bexport\b",
        r"\bdownload\b.*\bdata\b",
        r"\bworkspace\s+data\b",
        r"\bbackup\b",
    ],
}


def normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        text.lower().strip(),
    )


def classify_ticket(text):
    normalized = normalize_text(text)

    scores = {}
    matches = {}

    for category, patterns in CATEGORY_PATTERNS.items():
        category_matches = []

        for pattern in patterns:
            if re.search(pattern, normalized):
                category_matches.append(pattern)

        scores[category] = len(category_matches)
        matches[category] = category_matches

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_category, best_score = ranked[0]
    second_score = ranked[1][1]

    if best_score == 0:
        return {
            "category": "unknown",
            "confidence": 0.0,
            "matched_patterns": [],
        }

    if best_score == second_score:
        confidence = 0.5
    else:
        confidence = min(
            0.95,
            0.60 + (0.15 * best_score),
        )

        if second_score > 0:
            confidence -= 0.10 * (
                second_score / best_score
            )

    return {
        "category": best_category,
        "confidence": round(float(confidence), 3),
        "matched_patterns": matches[best_category],
    }
