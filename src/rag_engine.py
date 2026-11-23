from src.retriever import MedicalRetriever
from src.llm_generator import generate_answer


class MedicalRAGEngine:

    def __init__(self):
        self.retriever = MedicalRetriever()

    def ask_question(self, question):

        # Retrieve relevant passages
        retrieved_passages = self.retriever.search(
            question,
            top_k=3
        )

        # Generate answer using Groq
        answer = generate_answer(
            question,
            retrieved_passages
        )

        # Collect source information
        sources = []

        for passage in retrieved_passages:
            sources.append({
                "source": passage["source"],
                "page": passage["page"],
                "similarity_score": round(
                    passage["similarity_score"], 3
                )
            })

        return {
            "answer": answer,
            "sources": sources
        }