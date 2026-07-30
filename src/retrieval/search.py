from .embeddings import embed_text_clip, embed_text_dense


def reciprocal_rank_fusion(result_lists, k=60):
    """Combines multiple ranked result lists into one fused ranking."""
    scores, payloads = {}, {}
    for results in result_lists:
        for rank, point in enumerate(results):
            scores[point.id] = scores.get(point.id, 0) + 1 / (k + rank + 1)
            payloads[point.id] = point.payload
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(pid, payloads[pid], score) for pid, score in fused]


def hybrid_search(query_text, qdrant_client, clip_model, clip_processor, text_model,
                   device, image_collection, text_collection, prefetch_limit=10, final_k=5):
    """Runs both CLIP-based and text-dense search, fuses results with RRF,
    and returns only the top final_k fused candidates (not every unique
    page seen across both searches)."""
    clip_vector = embed_text_clip(query_text, clip_model, clip_processor, device)
    image_results = qdrant_client.query_points(
        collection_name=image_collection, query=clip_vector, limit=prefetch_limit
    ).points

    text_vector = embed_text_dense(query_text, text_model)
    text_results = qdrant_client.query_points(
        collection_name=text_collection, query=text_vector, limit=prefetch_limit
    ).points

    fused = reciprocal_rank_fusion([image_results, text_results])
    return fused[:final_k]
