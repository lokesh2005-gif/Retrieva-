import shutil
from pathlib import Path

from config import config


def test_ingest_uses_configured_chroma_directory():
    from ingest import ingest_pdfs

    assert ingest_pdfs.__defaults__[1] == config.CHROMA_PERSIST_DIRECTORY


def test_clear_chroma_db_removes_persist_directory(tmp_path):
    from ingest import clear_chroma_db

    target = tmp_path / "chroma_db"
    target.mkdir()
    (target / "chroma.sqlite3").write_text("dummy")

    clear_chroma_db(str(target))

    assert not target.exists()
