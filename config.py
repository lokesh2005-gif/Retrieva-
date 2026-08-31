import os


class Config:
    """Configuration for the PDF-only RAG application."""

    PDF_SOURCE_DIRECTORY: str = "data"
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma_db"

    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. Add PDF files in '{self.PDF_SOURCE_DIRECTORY}'.")


config = Config()
