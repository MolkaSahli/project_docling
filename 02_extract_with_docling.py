from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfe_ews.config import Settings  # noqa: E402
from pfe_ews.hybrid_extractor import extract_documents  # noqa: E402
from pfe_ews.logging_utils import configure_logging  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    configure_logging(settings.logs_dir, "02_extract_hybrid")
    rows = extract_documents(settings)
    summary = {
        "documents": len(rows),
        "succes": sum(row["extraction_status"] in {"success", "skipped_existing"} for row in rows),
        "erreurs": sum(row["extraction_status"] == "error" for row in rows),
        "tables_detectees": sum(int(row.get("num_tables") or 0) for row in rows),
        "tables_docling": sum(int(row.get("num_docling_tables") or 0) for row in rows),
        "tables_pymupdf": sum(int(row.get("num_pymupdf_tables") or 0) for row in rows),
        "images_decrites": sum(int(row.get("num_visuals") or 0) for row in rows),
        "manifest": str(settings.extracted_dir / "extraction_manifest.jsonl"),
        "controle_humain": "Ouvrir les .merged.md et .canonical.jsonl dans data/extracted/ et comparer avec les sources.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
