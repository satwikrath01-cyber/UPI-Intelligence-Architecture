"""
ingest.py — PDF ingestion with smart OCR fallback.
- Digital PDFs: fast text extraction via PyMuPDF
- Scanned PDFs: per-page OCR via Tesseract (auto-detected)
Bulk run: python ingest.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

CIRCULARS_DIR = Path(r"C:\Users\rathb\Downloads\NPCI\UPI Circulars\Combined")
CHROMA_DIR    = "chroma_db"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 180

# Common Tesseract install paths on Windows
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\rathb\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
]


# ── OCR setup ──────────────────────────────────────────────────────────────────

def _setup_tesseract() -> bool:
    """Locate Tesseract and configure pytesseract. Returns True if available."""
    try:
        import pytesseract
        for path in _TESSERACT_PATHS:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return True
        # Try system PATH
        import shutil
        if shutil.which("tesseract"):
            return True
        return False
    except ImportError:
        return False


def tesseract_available() -> bool:
    return _setup_tesseract()


# ── Smart PDF loader ───────────────────────────────────────────────────────────

def load_pdf_smart(pdf_path: str, name: str) -> tuple[list[Document], int]:
    """
    Extract text from a PDF, falling back to OCR for scanned pages.
    Returns (list_of_Documents, ocr_page_count).
    Requires: pymupdf (pip install pymupdf)
    OCR requires: pytesseract + Tesseract installed on system.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    ocr_ready = _setup_tesseract()
    ocr_count = 0
    documents = []

    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        # Page is scanned / image-only — attempt OCR
        if len(text) < 30:
            if ocr_ready:
                try:
                    import pytesseract
                    from PIL import Image

                    # Render at 2x resolution for better accuracy
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = pytesseract.image_to_string(img, lang="eng")
                    ocr_count += 1
                except Exception as e:
                    text = f"[OCR failed for page {page_num + 1}: {e}]"
            else:
                text = ""  # OCR not available; skip blank page

        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": name, "page": page_num + 1, "ocr": ocr_count > 0},
            ))

    doc.close()
    return documents, ocr_count


# ── Shared helpers ─────────────────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings: HuggingFaceEmbeddings = None) -> Chroma:
    if embeddings is None:
        embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="upi_circulars",
    )


def _chunk(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_pdf(pdf_path: str, vectorstore: Chroma) -> dict:
    """
    Ingest a single PDF (digital or scanned) into the vectorstore.
    Returns status dict with keys: status, chunks, pages, ocr_pages, name.
    """
    name = Path(pdf_path).name

    existing = vectorstore.get(where={"source": {"$eq": name}})
    if existing["ids"]:
        return {"status": "duplicate", "chunks": 0, "pages": 0, "ocr_pages": 0, "name": name}

    pages, ocr_pages = load_pdf_smart(pdf_path, name)

    chunks = [c for c in _chunk(pages) if c.page_content.strip()]
    if not chunks:
        return {"status": "empty", "chunks": 0, "pages": len(pages), "ocr_pages": ocr_pages, "name": name}

    vectorstore.add_documents(chunks)
    try:
        vectorstore.persist()
    except Exception:
        pass

    return {
        "status":    "added",
        "chunks":    len(chunks),
        "pages":     len(pages),
        "ocr_pages": ocr_pages,
        "name":      name,
    }


def list_circulars(vectorstore: Chroma) -> list[str]:
    result  = vectorstore.get(include=["metadatas"])
    sources = {m.get("source", "") for m in result["metadatas"] if m.get("source")}
    return sorted(sources)


# ── Bulk ingestion ─────────────────────────────────────────────────────────────

def _bulk_ingest(folder: Path):
    import shutil

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in '{folder}'.")
        sys.exit(1)

    # Wipe existing store so we get a clean rebuild with no duplicates
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"Cleared old vector store — rebuilding from scratch.\n")

    ocr_ok = tesseract_available()
    print(f"OCR available : {'YES — Tesseract found' if ocr_ok else 'NO  — install Tesseract for scanned PDFs'}")
    print(f"Circulars found: {len(pdfs)}")
    print(f"Embedding model: {EMBED_MODEL}\n")
    embeddings = get_embeddings()

    all_chunks = []
    total_ocr  = 0
    failed     = []

    for i, pdf in enumerate(pdfs, 1):
        print(f"  [{i:>3}/{len(pdfs)}] {pdf.name}", end="", flush=True)
        try:
            pages, ocr_n = load_pdf_smart(str(pdf), pdf.name)
            chunks = [c for c in _chunk(pages) if c.page_content.strip()]
            all_chunks.extend(chunks)
            total_ocr += ocr_n
            tag = f"  ({ocr_n} OCR pages)" if ocr_n else ""
            print(f"  → {len(chunks)} chunks{tag}")
        except Exception as e:
            print(f"  SKIPPED: {e}")
            failed.append(pdf.name)

    if not all_chunks:
        print("No text extracted from any PDF. Aborting.")
        sys.exit(1)

    print(f"\nEmbedding {len(all_chunks)} chunks across {len(pdfs) - len(failed)} circulars...")
    vs = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="upi_circulars",
    )
    try:
        vs.persist()
    except Exception:
        pass

    print(f"\n{'='*55}")
    print(f"  Done!  {vs._collection.count()} vectors stored in '{CHROMA_DIR}'")
    print(f"  Circulars loaded : {len(pdfs) - len(failed)} / {len(pdfs)}")
    if total_ocr:
        print(f"  OCR pages        : {total_ocr}")
    if failed:
        print(f"  Failed (skipped) : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    print("=== UPI Intelligence Architecture — Bulk Ingestion ===\n")
    _bulk_ingest(CIRCULARS_DIR)
    print("\nIngestion complete. Run 'streamlit run app.py' to start.")
