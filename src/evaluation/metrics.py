import re
import numpy as np


def compute_precision_recall(retrieved_docs, relevant_docs):
    retrieved_docs = set(retrieved_docs)
    relevant_docs = set(relevant_docs)
    if not relevant_docs:
        precision = 1.0 if not retrieved_docs else 0.0
        return precision, None
    if not retrieved_docs:
        return 0.0, 0.0
    overlap = len(retrieved_docs & relevant_docs)
    return overlap / len(retrieved_docs), overlap / len(relevant_docs)


def compute_answer_relevancy(query, answer, text_model):
    if not answer or not answer.strip():
        return 0.0
    query_vec = text_model.encode(query)
    answer_vec = text_model.encode(answer)
    return float(np.dot(query_vec, answer_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(answer_vec)))


def judge_faithfulness(answer, evidence_texts, gemini_client, model="gemini-flash-lite-latest"):
    if not evidence_texts:
        evidence_block = "(no evidence was provided to this system for this answer)"
    else:
        sources = evidence_texts[:5]
        per_source_budget = 8000 // len(sources)
        evidence_block = "\n\n".join(t[:per_source_budget] for t in sources)

    prompt = ("Rate 0-100 whether every claim in this answer is supported by the evidence. "
              "Respond with ONLY the number.\n\nEvidence:\n" + evidence_block +
              "\n\nAnswer:\n" + answer)

    response = gemini_client.models.generate_content(model=model, contents=prompt)
    match = re.search(r"\d+", response.text.strip())
    return min(int(match.group()), 100) if match else 0
