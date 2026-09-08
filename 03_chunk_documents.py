from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfe_ews.canonical_chunking import build_chunks  # noqa: E402
from pfe_ews.config import Settings  # noqa: E402
from pfe_ews.logging_utils import configure_logging  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.logs_dir, "03_chunk_documents")
    chunks = build_chunks(settings)
    summary = {
        "chunks": len(chunks),
        "chunks_tableaux": sum(bool(row.get("is_table")) for row in chunks),
        "contreparties": sorted({row["counterparty"] for row in chunks}),
        "sortie": str(settings.chunks_dir / "chunks.jsonl"),
        "taille_cible_tokens": settings.chunk_max_tokens,
        "overlap_tokens": settings.chunk_overlap_tokens,
        "chunking_texte": "RecursiveCharacterTextSplitter",
        "chunking_tables": "row_splitter_with_repeated_header",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
