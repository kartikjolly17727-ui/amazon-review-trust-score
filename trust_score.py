from model import FAKE_LABEL, REAL_LABEL


def _trust_from_probability(real_probability: float) -> int:
    score = 100 * (real_probability**0.65)
    return max(0, min(100, round(score)))


def build_trust_score(prediction: dict[str, object]) -> dict[str, object]:
    real_probability = float(prediction["real_probability"])
    fake_probability = float(prediction["fake_probability"])
    label = str(prediction["label"])

    score = _trust_from_probability(real_probability)

    if label == REAL_LABEL and score >= 80:
        verdict = "High trust"
        summary = "This review looks likely to be written by a real customer."
    elif label == REAL_LABEL:
        verdict = "Medium trust"
        summary = "This review leans real, with some uncertainty."
    elif fake_probability >= 0.75:
        verdict = "Low trust"
        summary = "This review has strong signs of being computer generated."
    else:
        verdict = "Needs review"
        summary = "This review leans fake, but the model confidence is moderate."

    return {
        "score": score,
        "verdict": verdict,
        "summary": summary,
        "real_percent": round(real_probability * 100, 1),
        "fake_percent": round(fake_probability * 100, 1),
    }
