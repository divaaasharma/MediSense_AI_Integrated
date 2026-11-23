import re
import json
from pathlib import Path
from pypdf import PdfReader

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_DIR / "data" / "medical_documents"
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "processed_documents.json"


def clean_text(text):
    """Clean extracted PDF text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\S ]+", " ", text)
    return text.strip()


def load_documents():
    """Load PDF files and extract text page by page."""
    documents = []

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found!")
        return documents

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))

            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    documents.append({
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": clean_text(text)
                    })

        except Exception as error:
            print(f"Error processing {pdf_path.name}: {error}")

    return documents


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(documents, file, indent=4, ensure_ascii=False)

    print("\nDocument processing completed!")
    print(f"Total pages extracted: {len(documents)}")
    print(f"Saved to: {OUTPUT_FILE}")