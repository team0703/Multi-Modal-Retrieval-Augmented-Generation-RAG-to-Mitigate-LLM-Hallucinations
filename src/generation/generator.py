import re
from PIL import Image


def generate_grounded_response(query, verified_evidence, gemini_client, model="gemini-flash-lite-latest"):
    """
    Generates an answer grounded in the verified evidence from Phase 3, with
    per-claim citations and a blended confidence score (self-reported model
    confidence averaged with the evidence's own verification strength).
    Returns a safe "can't answer" response if no evidence passed verification,
    rather than letting the model answer from general knowledge.
    """
    if not verified_evidence:
        return {
            "answer": "I don't have enough verified evidence to answer this question confidently.",
            "citations": [],
            "confidence": 0
        }

    evidence_text = ""
    images = []
    for i, ev in enumerate(verified_evidence):
        evidence_text += f"\n[Source {i+1}: {ev['doc_id']}, page {ev['page_num']}]\n{ev['text'][:1500]}\n"
        images.append(Image.open(ev["image_path"]))

    prompt = f"""Answer the question using ONLY the evidence provided below. For every claim in your answer, cite the source using the format [Source N] matching the sources listed.

Question: {query}

Evidence:
{evidence_text}

After your answer, on a new line, add "CONFIDENCE: X" where X is a number from 0-100 representing how confident you are that your answer is fully supported by the evidence above (not general knowledge). Be honest -- if the evidence only partially answers the question, reflect that with a lower number."""

    response = gemini_client.models.generate_content(
        model=model,
        contents=[prompt] + images
    )

    response_text = response.text.strip()

    self_reported_confidence = 50
    match = re.search(r"CONFIDENCE:\s*(\d+)", response_text)
    if match:
        self_reported_confidence = min(int(match.group(1)), 100)
        answer_text = response_text[:match.start()].strip()
    else:
        answer_text = response_text

    avg_verification_score = sum(ev["verification_score"] for ev in verified_evidence) / len(verified_evidence)
    blended_confidence = round((self_reported_confidence + avg_verification_score) / 2)

    citations = [{"doc_id": ev["doc_id"], "page_num": ev["page_num"]} for ev in verified_evidence]

    return {
        "answer": answer_text,
        "citations": citations,
        "confidence": blended_confidence,
        "self_reported_confidence": self_reported_confidence,
        "avg_verification_score": round(avg_verification_score, 1)
    }
