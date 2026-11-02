
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data" / "processed" / "text_chunks.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks():
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def main():
    chunks = load_chunks()

    if not chunks:
        print("No text chunks found.")
        return

    texts = [
        f"{chunk['title']}\n{chunk['content']}"
        for chunk in chunks
    ]

    print(f"Loaded {len(chunks)} text chunks.")
    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print("Semantic retrieval is ready.")

    while True:
        query = input(
            "\nAsk about traffic advisories or weather "
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
        top_indices = scores.argsort(descending=True)[:3]

        print("\nTop retrieved passages:")

        for index in top_indices:
            chunk = chunks[int(index)]

            print(f"\nTitle: {chunk['title']}")
            print(f"Source: {chunk['source_file']}")
            print(f"Section: {chunk['metadata'].get('section', 'N/A')}")
            print(f"Similarity: {float(scores[index]):.3f}")
            print(f"Passage: {chunk['content'][:600]}")

    print("Chunk retrieval test completed.")


if __name__ == "__main__":
    main()
