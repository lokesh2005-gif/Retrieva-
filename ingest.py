import os
import shutil
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import config

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_pdf_content(pdf_directory: str):
    """Load all PDF documents from a directory."""

    print(f"Starting PDF document loading from '{pdf_directory}'...")

    if not os.path.exists(pdf_directory):
        print(f"Error: PDF directory '{pdf_directory}' not found.")
        print("Please create this directory and place PDF files inside.")
        return []

    all_pdf_docs = []

    for filename in sorted(os.listdir(pdf_directory)):

        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(pdf_directory, filename)

        print(f"Loading PDF document: {filepath}")

        try:
            loader = PyPDFLoader(filepath)
            pages = loader.load()

            all_pdf_docs.extend(pages)

            print(f"Loaded {len(pages)} pages from {filename}")

        except Exception as exc:
            print(f"Error loading {filepath}: {exc}")

    if not all_pdf_docs:
        print("No PDF documents found in the specified directory.")
    else:
        print(f"Loaded {len(all_pdf_docs)} total pages from PDF documents.")

    return all_pdf_docs


def clear_chroma_db(persist_directory: str | None = None):
    """Remove the persisted Chroma database."""

    target = persist_directory or config.CHROMA_PERSIST_DIRECTORY

    if not os.path.exists(target):
        print(
            f"Chroma DB directory '{target}' "
            "does not exist. Nothing to clear."
        )
        return

    try:
        if os.path.isdir(target):
            shutil.rmtree(target)
        else:
            os.remove(target)

        print(f"Cleared Chroma database at '{target}'.")

    except Exception as exc:
        print(f"Error clearing Chroma database: {exc}")


def ingest_pdfs(
    pdf_directory: str = "data",
    persist_directory: str = config.CHROMA_PERSIST_DIRECTORY,
):
    """Load PDFs, split them into chunks, embed them, and store them in Chroma."""

    print("\n" + "=" * 60)
    print("STARTING PDF INGESTION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load PDF documents
    # ---------------------------------------------------------

    pdf_docs = load_pdf_content(pdf_directory)

    if not pdf_docs:
        print("No PDF documents were loaded.")
        print("Ingestion stopped.")
        return

    print(f"\nTotal PDF pages loaded: {len(pdf_docs)}")

    # ---------------------------------------------------------
    # 2. Split documents into chunks
    # ---------------------------------------------------------

    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP

    print("\nSplitting documents...")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunked_docs = text_splitter.split_documents(pdf_docs)

    print(f"Created {len(chunked_docs)} document chunks.")

    if not chunked_docs:
        print("No chunks were created.")
        return

    # ---------------------------------------------------------
    # 3. Initialize embedding model
    # ---------------------------------------------------------

    model_name = config.EMBEDDING_MODEL_NAME

    print("\nInitializing embedding model...")
    print(f"Embedding model: {model_name}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

        print("Embedding model initialized successfully.")

    except Exception as exc:
        print(f"Failed to initialize embedding model: {exc}")
        return

    # ---------------------------------------------------------
    # 4. Prepare Chroma directory
    # ---------------------------------------------------------

    os.makedirs(persist_directory, exist_ok=True)

    print("\nChroma database location:")
    print(persist_directory)

    # ---------------------------------------------------------
    # 5. Remove old collection if required
    # ---------------------------------------------------------

    print("\nCreating Chroma vector database...")

    try:
        vectordb = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name="pdf_documents",
        )

    except Exception as exc:

        print("\nERROR: Failed to create Chroma database.")
        print(exc)

        print("\nIf this directory contains an old Chroma database,")
        print("delete the existing Chroma directory and run ingestion again.")

        return

    # ---------------------------------------------------------
    # 6. Verify database
    # ---------------------------------------------------------

    try:
        collection_count = vectordb._collection.count()

        print("\nChroma database created successfully.")
        print(f"Documents stored in Chroma: {collection_count}")

    except Exception:
        print("Chroma database created successfully.")

    # ---------------------------------------------------------
    # 7. Completion
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PDF INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"PDF directory : {pdf_directory}")
    print(f"Chroma DB     : {persist_directory}")
    print(f"Chunks        : {len(chunked_docs)}")
    print("Collection    : pdf_documents")
    print("=" * 60)


if __name__ == "__main__":

    ingest_pdfs(
        pdf_directory=config.PDF_SOURCE_DIRECTORY,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    )