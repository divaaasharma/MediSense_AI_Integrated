import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = PROJECT_DIR / "data" / "vector_store"

INDEX_FILE = VECTOR_DIR / "medical_index.faiss"
METADATA_FILE = VECTOR_DIR / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class MedicalRetriever:

    def __init__(self):
        print("Loading FAISS database...")

        self.index = faiss.read_index(str(INDEX_FILE))

        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.model = SentenceTransformer(MODEL_NAME)

        print("Medical retriever ready!")

    def search(self, question, top_k=3):
        """Retrieve the most relevant medical passages."""

        query_embedding = self.model.encode(
            [question],
            normalize_embeddings=True
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(distances[0], indices[0]):

            if idx == -1:
                continue

            document = self.metadata[idx]

            results.append({
                "source": document["source"],
                "page": document["page"],
                "text": document["text"],
                "similarity_score": float(score)
            })

        return results


if __name__ == "__main__":

    retriever = MedicalRetriever()

    question = input("\nEnter your medical question: ")

    results = retriever.search(question)

    print("\nRelevant passages:\n")

    for result in results:
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Similarity:", round(result["similarity_score"], 3))
        print("Content:", result["text"][:500])
        print("-" * 60)