import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Load API key from .env
load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("Groq API key not found. Check your .env file.")

# Initialize Groq client
client = Groq(api_key=API_KEY)


def generate_answer(question, retrieved_passages):
    """
    Generate an answer using retrieved medical documents.
    """

    if not retrieved_passages:
        return "The answer is not available in the provided medical documents."

    # Prepare document context
    context = ""

    for passage in retrieved_passages:
        context += (
            f"\nSource: {passage['source']}, "
            f"Page: {passage['page']}\n"
            f"Content: {passage['text']}\n"
        )

    # Create grounded prompt
    prompt = f"""
You are an evidence-based medical information assistant.

Answer the user's question using ONLY the medical document
passages provided below.

STRICT RULES:
1. Do not use information outside the provided passages.
2. Do not invent medical facts.
3. If the answer is not supported by the passages, respond
   exactly with:
   The answer is not available in the provided medical documents.
4. Do not include source citations in your generated answer.
5. Explain the answer in simple language.
6. Do not diagnose diseases or prescribe medication.
7. Do not provide unsupported medical advice.

MEDICAL DOCUMENT PASSAGES:
{context}

USER QUESTION:
{question}
"""

    # Generate response using Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful medical information assistant. "
                    "Your answers must be strictly grounded in the "
                    "provided healthcare documents."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    answer = response.choices[0].message.content.strip()

    # Handle unavailable answers
    unavailable_message = (
        "The answer is not available in the provided medical documents."
    )

    if unavailable_message.lower() in answer.lower():
        return unavailable_message

    return answer


if __name__ == "__main__":
    print("LLM generator is ready.")