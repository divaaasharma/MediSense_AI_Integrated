import json
from pathlib import Path

# File paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed_documents.json"
OUTPUT_FILE = PROJECT_DIR / "data" / "document_chunks.json"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])

        if chunk_text.strip():
            chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def process_documents():
    """Create chunks while preserving source and page metadata."""
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    all_chunks = []

    for document in documents:
        text_chunks = create_chunks(document["text"])

        for index, chunk in enumerate(text_chunks):
            all_chunks.append({
                "source": document["source"],
                "page": document["page"],
                "chunk_id": index,
                "text": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = process_documents()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print("Text chunking completed!")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")