import re
import time
from PIL import Image


def verify_evidence(query, candidate_text, candidate_image_path, gemini_client, model="gemini-flash-lite-latest", max_retries=3):
    """
    Uses Gemini to score how relevant a retrieved page is to a query.
    Looks at both the extracted text and the actual page image.
    Retries with backoff on rate-limit (429) errors, since the free tier
    allows only a few requests per minute. Returns 0 (fails closed) if the
    response can't be parsed or retries are exhausted.
    """
    image = Image.open(candidate_image_path)

    prompt = f"""You are checking whether a retrieved document page is relevant and useful for answering a question.

Question: {query}

Extracted text from this page (may be incomplete):
{candidate_text[:2000]}

Look at both the text above and the attached page image (which may contain charts, tables, or diagrams not captured in the text). Rate how relevant and useful this page is for answering the question, from 0 to 100:
- 0-20: Not relevant
- 21-50: Marginally related, doesn't help answer the question
- 51-80: Relevant, partially useful
- 81-100: Highly relevant, directly useful

Respond with ONLY the number. No words, no explanation."""

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=model,
                contents=[prompt, image]
            )
            match = re.search(r"\d+", response.text.strip())
            if match:
                return min(int(match.group()), 100)
            return 0
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait_time = 20 * (attempt + 1)
                print(f"    Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise

    print("    Gave up after max retries, scoring as 0 (not verified)")
    return 0


def verify_candidates(query, candidates, gemini_client, threshold=50, model="gemini-flash-lite-latest", delay=13):
    """
    Runs verify_evidence on each (id, payload, retrieval_score) candidate from
    hybrid_search, pausing `delay` seconds between calls. Returns (passed, all_verified).
    """
    verified = []
    for i, (pid, payload, retrieval_score) in enumerate(candidates):
        if i > 0:
            time.sleep(delay)
        score = verify_evidence(query, payload["text"], payload["image_path"], gemini_client, model)
        verified.append({
            "doc_id": payload["doc_id"],
            "page_num": payload["page_num"],
            "text": payload["text"],
            "image_path": payload["image_path"],
            "retrieval_score": retrieval_score,
            "verification_score": score
        })

    passed = [v for v in verified if v["verification_score"] >= threshold]
    passed.sort(key=lambda v: v["verification_score"], reverse=True)
    return passed, verified
