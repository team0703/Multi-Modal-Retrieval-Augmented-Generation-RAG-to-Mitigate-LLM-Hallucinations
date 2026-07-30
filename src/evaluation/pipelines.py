from ..retrieval.search import hybrid_search
from ..verification.verifier import verify_candidates
from ..generation.generator import generate_grounded_response


def baseline_answer(query, gemini_client, model="gemini-flash-lite-latest"):
    """No-RAG baseline: asks the model directly with no retrieved context at all."""
    response = gemini_client.models.generate_content(
        model=model,
        contents=query
    )
    return {
        "answer": response.text.strip(),
        "citations": [],
        "pipeline": "baseline_no_rag"
    }


def text_only_rag_answer(query, qdrant_client, text_model, gemini_client, text_collection,
                          limit=5, model="gemini-flash-lite-latest"):
    """Standard text-only RAG: retrieves using only the text-dense embeddings,
    no visual retrieval, no verification step -- a simpler comparison baseline."""
    text_vector = text_model.encode(query).tolist()
    results = qdrant_client.query_points(
        collection_name=text_collection, query=text_vector, limit=limit
    ).points

    if not results:
        return {"answer": "No relevant documents found.", "citations": [], "pipeline": "text_only_rag"}

    context = ""
    citations = []
    for i, point in enumerate(results):
        context += f"\n[Source {i+1}: {point.payload['doc_id']}, page {point.payload['page_num']}]\n{point.payload['text'][:1500]}\n"
        citations.append({"doc_id": point.payload["doc_id"], "page_num": point.payload["page_num"]})

    prompt = f"""Answer the question using the following retrieved text passages. Cite sources using [Source N] format.

Question: {query}

Retrieved passages:
{context}"""

    response = gemini_client.models.generate_content(model=model, contents=prompt)

    return {
        "answer": response.text.strip(),
        "citations": citations,
        "pipeline": "text_only_rag"
    }


def mm_rag_answer(query, qdrant_client, clip_model, clip_processor, text_model, device,
                   image_collection, text_collection, gemini_client, threshold=50,
                   model="gemini-flash-lite-latest"):
    """Full MM-RAG: hybrid retrieval + evidence verification + grounded generation."""
    candidates = hybrid_search(
        query, qdrant_client, clip_model, clip_processor, text_model, device,
        image_collection, text_collection
    )
    passed, all_verified = verify_candidates(query, candidates, gemini_client, threshold, model)
    result = generate_grounded_response(query, passed, gemini_client, model)
    result["pipeline"] = "mm_rag"
    result["num_candidates"] = len(candidates)
    result["num_passed_verification"] = len(passed)
    return result
