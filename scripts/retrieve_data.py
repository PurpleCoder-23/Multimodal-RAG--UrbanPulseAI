
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "ingested_records.jsonl"

# Downloaded Sentence Transformers model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    documents = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)

            # Semantic retrieval for descriptive text.
            # Roads and traffic will also use structured retrieval.
            if document["source_type"] in {"advisory", "weather"}:
                documents.append(document)

    return documents


def main():
    documents = load_documents()

    if not documents:
        print("No documents available for semantic retrieval.")
        return

    texts = [
        f"{doc['title']}\n{doc['content']}"
        for doc in documents
    ]

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    while True:
        query = input(
            "\nAsk a question about the advisory or weather "
            "(type 'exit' to stop): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        query_embedding = model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        scores = util.cos_sim(query_embedding, embeddings)[0]
        top_indices = scores.argsort(descending=True)[:2]

        print("\nRetrieved evidence:")

        for index in top_indices:
            doc = documents[int(index)]

            print(f"\nSource: {doc['source_file']}")
            print(f"Type: {doc['source_type']}")
            print(f"Title: {doc['title']}")
            print(f"Similarity: {float(scores[index]):.3f}")
            print(f"Content preview: {doc['content'][:400]}")

    print("Semantic retrieval test completed.")


if __name__ == "__main__":
    main()
