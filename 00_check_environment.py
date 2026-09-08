from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfe_ews.config import Settings  # noqa: E402
from pfe_ews.embedding_profiles import model_choice_explanation  # noqa: E402
from pfe_ews.enterprise_api import (  # noqa: E402
    EnterpriseChatClient,
    EnterpriseEmbeddingClient,
)
from pfe_ews.io_utils import write_json  # noqa: E402
from pfe_ews.logging_utils import configure_logging  # noqa: E402


def package_status(distribution: str, module: str | None = None) -> dict[str, object]:
    module_name = module or distribution.replace("-", "_")
    available = importlib.util.find_spec(module_name) is not None
    try:
        installed_version = version(distribution)
    except PackageNotFoundError:
        installed_version = None
    return {
        "distribution": distribution,
        "module": module_name,
        "available": available,
        "version": installed_version,
    }


def safe_endpoint(value: str) -> str | None:
    if not value:
        return None
    parsed = urlparse(value)
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "configured"


def main() -> None:
    settings = Settings.from_env()
    settings.ensure_directories()
    log_path = configure_logging(settings.logs_dir, "00_check_environment")

    supported = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".csv"}
    input_files = [
        path
        for path in settings.input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    ]
    libreoffice = settings.libreoffice_path or shutil.which("soffice") or shutil.which("libreoffice")

    packages = [
        package_status("docling"),
        package_status("docling-core", "docling_core"),
        package_status("pymupdf4llm"),
        package_status("PyMuPDF", "pymupdf"),
        package_status("qdrant-client", "qdrant_client"),
        package_status("httpx"),
        package_status("numpy"),
        package_status("openpyxl"),
        package_status("pydantic"),
        package_status("reportlab"),
        package_status("rapidfuzz"),
    ]

    report: dict[str, object] = {
        "python": sys.version,
        "project_root": str(PROJECT_ROOT),
        "configuration": settings.configuration_summary(),
        "embedding_choice": model_choice_explanation(settings.embedding_model),
        "packages": packages,
        "input_document_count": len(input_files),
        "input_documents": [str(path.relative_to(settings.input_dir)) for path in input_files],
        "libreoffice": libreoffice,
        "embedding_endpoint": safe_endpoint(settings.embedding_base_url),
        "embedding_key_present": bool(settings.embedding_api_key),
        "llm_endpoint": safe_endpoint(settings.llm_base_url),
        "llm_key_present": bool(settings.llm_api_key),
        "qdrant_mode": "embedded_local_no_server_no_qdrant_api_key",
        "log_path": str(log_path),
        "warnings": [],
        "connectivity": {},
    }

    warnings = report["warnings"]
    assert isinstance(warnings, list)
    missing = [item["distribution"] for item in packages if not item["available"]]
    if missing:
        warnings.append("Packages absents: " + ", ".join(str(item) for item in missing))
    if any(path.suffix.lower() in {".doc", ".docx", ".xls"} for path in input_files) and not libreoffice:
        warnings.append("LibreOffice est requis/recommande pour convertir Word et les anciens .xls.")
    if not settings.embedding_base_url:
        warnings.append("EMBEDDING_BASE_URL n'est pas renseignee dans .env.")
    if not settings.llm_base_url:
        warnings.append("LLM_BASE_URL n'est pas renseignee dans .env.")
    if settings.docling_artifacts_path and not settings.docling_artifacts_path.exists():
        warnings.append("DOCLING_ARTIFACTS_PATH pointe vers un dossier inexistant.")

    if settings.run_connectivity_tests:
        connectivity: dict[str, object] = {}
        try:
            matrix = EnterpriseEmbeddingClient(settings).embed(["test de connectivite sans donnees bancaires"])
            connectivity["embedding"] = {
                "ok": True,
                "shape": list(matrix.shape),
            }
        except Exception as exc:
            connectivity["embedding"] = {"ok": False, "error": str(exc)}
        try:
            text, _ = EnterpriseChatClient(settings).complete(
                system_prompt="Retourne uniquement un JSON valide.",
                user_prompt='Retourne {"ok": true}.',
            )
            connectivity["llm"] = {"ok": True, "preview": text[:200]}
        except Exception as exc:
            connectivity["llm"] = {"ok": False, "error": str(exc)}
        report["connectivity"] = connectivity

    output = settings.logs_dir / "environment_check.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nRapport technique: {output}")


if __name__ == "__main__":
    main()
