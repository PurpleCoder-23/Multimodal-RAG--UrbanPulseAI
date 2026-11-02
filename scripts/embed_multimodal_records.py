
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multimodal_records.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

EMBEDDINGS_PATH = OUTPUT_DIR / "multimodal_embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "multimodal_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load multimodal evidence records
    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("No multimodal records found.")

    texts = [record["content"] for record in records]

    print(f"Records loaded: {len(records)}")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Save vectors
    np.save(EMBEDDINGS_PATH, embeddings)

    # Save corresponding records and metadata
    metadata = [
        {
            "record_id": record["record_id"],
            "modality": record["modality"],
            "evidence_type": record["evidence_type"],
            "content": record["content"],
            "metadata": record["metadata"],
        }
        for record in records
    ]

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nEmbedding index created successfully.")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved: {EMBEDDINGS_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")


if __name__ == "__main__":
    main()
