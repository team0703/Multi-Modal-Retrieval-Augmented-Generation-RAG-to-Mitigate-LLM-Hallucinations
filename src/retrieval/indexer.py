from qdrant_client.models import Distance, VectorParams, PointStruct


def create_collection(qdrant_client, collection_name, vector_size):
    """Creates (or recreates) a Qdrant collection with the given vector size."""
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )


def upload_points(qdrant_client, collection_name, entries, embed_fn):
    """Embeds each entry with embed_fn and uploads to the given collection.
    embed_fn takes one entry dict and returns a vector (list of floats)."""
    points = []
    for idx, entry in enumerate(entries):
        vector = embed_fn(entry)
        points.append(PointStruct(
            id=idx,
            vector=vector,
            payload={
                "doc_id": entry["doc_id"],
                "page_num": entry["page_num"],
                "text": entry["text"],
                "image_path": entry["image_path"]
            }
        ))
    qdrant_client.upsert(collection_name=collection_name, points=points)
    return len(points)
