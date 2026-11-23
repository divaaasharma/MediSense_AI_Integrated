import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "document_chunks.json"
VECTOR_DIR = PROJECT_DIR / "data" / "vector_store"

INDEX_FILE = VECTOR_DIR / "medical_index.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.json"

# Lightweight embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_vector_store():
    """Generate embeddings and save them in FAISS."""

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        raise ValueError("No document chunks found!")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Create FAISS similarity index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save index and metadata
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print("\nFAISS vector database created successfully!")
    print(f"Total vectors stored: {index.ntotal}")
    print(f"Embedding dimensions: {dimension}")
    print(f"Saved index: {INDEX_FILE}")
    print(f"Saved metadata: {METADATA_FILE}")


if __name__ == "__main__":
    create_vector_store()