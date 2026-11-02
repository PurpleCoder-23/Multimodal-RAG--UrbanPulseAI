
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_PATH = PROCESSED_DIR / "multimodal_embeddings.npy"
METADATA_PATH = PROCESSED_DIR / "multimodal_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match.")

    return embeddings, metadata


def retrieve(query, model, embeddings, metadata, top_k=5):
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Cosine similarity because vectors are normalized
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        record = metadata[index]

        results.append({
            "record_id": record["record_id"],
            "modality": record["modality"],
            "evidence_type": record["evidence_type"],
            "content": record["content"],
            "source": record["metadata"]["source_image"],
            "similarity": float(scores[index]),
        })

    return results


def main():
    embeddings, metadata = load_index()

    print(f"Evidence records loaded: {len(metadata)}")
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("\nMultimodal evidence retrieval is ready.")
    print("Type 'exit' to stop.")

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = retrieve(
            query,
            model,
            embeddings,
            metadata,
            top_k=5,
        )

        print("\n--- Retrieved Evidence ---")

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Record: {result['record_id']}")
            print(f"Type: {result['evidence_type']}")
            print(f"Modality: {result['modality']}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Content: {result['content']}")
            print(f"Source: {result['source']}")


if __name__ == "__main__":
    main()
