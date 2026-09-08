from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on", "oui"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _env_json_object(name: str) -> dict[str, Any]:
    """Lit un objet JSON en conservant les types des valeurs."""

    raw = os.getenv(name, "{}").strip() or "{}"
    value: Any = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{name} doit etre un objet JSON.")
    return {str(key): item for key, item in value.items()}


def _env_json_string_object(name: str) -> dict[str, str]:
    """Lit un objet JSON destine aux en-tetes HTTP."""

    value = _env_json_object(name)
    return {str(key): str(item) for key, item in value.items()}


def _resolve_path(raw: str, default: str | None = None) -> Path | None:
    value = raw.strip() if raw else (default or "")
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    input_dir: Path
    prepared_dir: Path
    extracted_dir: Path
    chunks_dir: Path
    index_dir: Path
    outputs_dir: Path
    logs_dir: Path

    default_counterparty: str
    target_counterparty: str | None

    libreoffice_path: str | None
    require_word_to_pdf: bool
    allow_direct_word_fallback: bool

    docling_artifacts_path: Path | None
    docling_ocr_enabled: bool
    docling_ocr_languages: tuple[str, ...]
    docling_force_full_page_ocr: bool
    docling_do_cell_matching: bool
    docling_max_num_pages: int
    docling_max_file_size_mb: int
    force_reextract: bool

    pymupdf_ocr_language: str
    pymupdf_ocr_dpi: int
    pymupdf_table_strategy: str
    hybrid_table_overlap_threshold: float

    picture_description_enabled: bool
    picture_description_url: str
    picture_description_model: str | None
    picture_description_prompt: str
    picture_description_timeout_seconds: int
    picture_description_max_tokens: int
    picture_description_area_threshold: float
    picture_description_images_scale: float
    picture_description_extra_body: dict[str, Any]

    chunk_max_tokens: int
    chunk_overlap_tokens: int
    chunk_tokenizer_path: Path | None
    chunk_include_picture_ocr: bool

    embedding_base_url: str
    embedding_endpoint_path: str
    embedding_api_key: str
    embedding_auth_mode: str
    embedding_api_key_header: str
    embedding_model: str
    embedding_batch_size: int
    embedding_timeout_seconds: int
    embedding_dimensions: int | None
    embedding_query_prefix: str
    embedding_passage_prefix: str
    embedding_query_instruction: str
    embedding_extra_body: dict[str, Any]

    llm_base_url: str
    llm_endpoint_path: str
    llm_api_key: str
    llm_auth_mode: str
    llm_api_key_header: str
    llm_model: str
    llm_timeout_seconds: int
    llm_temperature: float
    llm_max_output_tokens: int
    llm_use_json_mode: bool
    llm_review_model: str | None
    llm_extra_body: dict[str, Any]

    tls_verify: bool
    ca_bundle_path: Path | None
    enterprise_extra_headers: dict[str, str]
    http_max_retries: int
    http_backoff_seconds: float

    vector_store_backend: str
    qdrant_local_path: Path
    qdrant_collection: str
    allow_numpy_fallback: bool
    save_embeddings_copy: bool

    retrieval_dense_top_k: int
    retrieval_lexical_top_k: int
    retrieval_final_top_k: int
    rrf_k: int
    max_retrieved_chunks_per_task: int

    min_evidence_fuzzy_score: int
    dedupe_similarity_threshold: int
    report_title: str
    report_confidentiality_label: str
    run_connectivity_tests: bool

    @classmethod
    def from_env(cls) -> "Settings":
        root = PROJECT_ROOT
        input_dir = _resolve_path("data/input")
        prepared_dir = _resolve_path("data/prepared")
        extracted_dir = _resolve_path("data/extracted")
        chunks_dir = _resolve_path("data/chunks")
        index_dir = _resolve_path("data/index")
        outputs_dir = _resolve_path("data/outputs")
        logs_dir = _resolve_path("data/logs")
        assert input_dir and prepared_dir and extracted_dir and chunks_dir
        assert index_dir and outputs_dir and logs_dir

        target = _env("TARGET_COUNTERPARTY") or None
        libreoffice = _env("LIBREOFFICE_PATH") or None
        languages = tuple(
            part.strip()
            for part in _env("DOCLING_OCR_LANGUAGES", "fr,en").split(",")
            if part.strip()
        )

        qdrant_path = _resolve_path(
            _env("QDRANT_LOCAL_PATH", "data/index/qdrant")
        )
        assert qdrant_path is not None

        instance = cls(
            project_root=root,
            input_dir=input_dir,
            prepared_dir=prepared_dir,
            extracted_dir=extracted_dir,
            chunks_dir=chunks_dir,
            index_dir=index_dir,
            outputs_dir=outputs_dir,
            logs_dir=logs_dir,
            default_counterparty=_env("DEFAULT_COUNTERPARTY", "CONTREPARTIE_INCONNUE"),
            target_counterparty=target,
            libreoffice_path=libreoffice,
            require_word_to_pdf=_env_bool("REQUIRE_WORD_TO_PDF", True),
            allow_direct_word_fallback=_env_bool("ALLOW_DIRECT_WORD_FALLBACK", True),
            docling_artifacts_path=_resolve_path(_env("DOCLING_ARTIFACTS_PATH")),
            docling_ocr_enabled=_env_bool("DOCLING_OCR_ENABLED", True),
            docling_ocr_languages=languages or ("fr", "en"),
            docling_force_full_page_ocr=_env_bool(
                "DOCLING_FORCE_FULL_PAGE_OCR", False
            ),
            docling_do_cell_matching=_env_bool("DOCLING_DO_CELL_MATCHING", True),
            docling_max_num_pages=_env_int("DOCLING_MAX_NUM_PAGES", 500),
            docling_max_file_size_mb=_env_int("DOCLING_MAX_FILE_SIZE_MB", 250),
            force_reextract=_env_bool("FORCE_REEXTRACT", False),
            pymupdf_ocr_language=_env("PYMUPDF_OCR_LANGUAGE", "fra+eng"),
            pymupdf_ocr_dpi=_env_int("PYMUPDF_OCR_DPI", 300),
            pymupdf_table_strategy=_env("PYMUPDF_TABLE_STRATEGY", "lines"),
            hybrid_table_overlap_threshold=_env_float(
                "HYBRID_TABLE_OVERLAP_THRESHOLD", 0.40
            ),
            picture_description_enabled=_env_bool(
                "PICTURE_DESCRIPTION_ENABLED", False
            ),
            picture_description_url=_env("PICTURE_DESCRIPTION_URL"),
            picture_description_model=_env("PICTURE_DESCRIPTION_MODEL") or None,
            picture_description_prompt=_env(
                "PICTURE_DESCRIPTION_PROMPT",
                "Describe this figure from a credit-risk document in 3 to 6 concise sentences. "
                "Preserve readable dates, labels, axes and material numerical values. "
                "For a chart, explain the trend and relationships. Do not invent unreadable values.",
            ),
            picture_description_timeout_seconds=_env_int(
                "PICTURE_DESCRIPTION_TIMEOUT_SECONDS", 120
            ),
            picture_description_max_tokens=_env_int(
                "PICTURE_DESCRIPTION_MAX_TOKENS", 400
            ),
            picture_description_area_threshold=_env_float(
                "PICTURE_DESCRIPTION_AREA_THRESHOLD", 0.02
            ),
            picture_description_images_scale=_env_float(
                "PICTURE_DESCRIPTION_IMAGES_SCALE", 2.0
            ),
            picture_description_extra_body=_env_json_object(
                "PICTURE_DESCRIPTION_EXTRA_BODY_JSON"
            ),
            chunk_max_tokens=_env_int("CHUNK_MAX_TOKENS", 700),
            chunk_overlap_tokens=_env_int("CHUNK_OVERLAP_TOKENS", 100),
            chunk_tokenizer_path=_resolve_path(_env("CHUNK_TOKENIZER_PATH")),
            chunk_include_picture_ocr=_env_bool(
                "CHUNK_INCLUDE_PICTURE_OCR", True
            ),
            embedding_base_url=_env("EMBEDDING_BASE_URL"),
            embedding_endpoint_path=_env("EMBEDDING_ENDPOINT_PATH", "/embeddings"),
            embedding_api_key=_env("EMBEDDING_API_KEY"),
            embedding_auth_mode=_env("EMBEDDING_AUTH_MODE", "bearer").lower(),
            embedding_api_key_header=_env(
                "EMBEDDING_API_KEY_HEADER", "X-API-Key"
            ),
            embedding_model=_env("EMBEDDING_MODEL", "bge-m3-ITG"),
            embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 16),
            embedding_timeout_seconds=_env_int(
                "EMBEDDING_TIMEOUT_SECONDS", 180
            ),
            embedding_dimensions=_env_optional_int("EMBEDDING_DIMENSIONS"),
            embedding_query_prefix=_env("EMBEDDING_QUERY_PREFIX"),
            embedding_passage_prefix=_env("EMBEDDING_PASSAGE_PREFIX"),
            embedding_query_instruction=_env("EMBEDDING_QUERY_INSTRUCTION"),
            embedding_extra_body=_env_json_object("EMBEDDING_EXTRA_BODY_JSON"),
            llm_base_url=_env("LLM_BASE_URL"),
            llm_endpoint_path=_env("LLM_ENDPOINT_PATH", "/chat/completions"),
            llm_api_key=_env("LLM_API_KEY"),
            llm_auth_mode=_env("LLM_AUTH_MODE", "bearer").lower(),
            llm_api_key_header=_env("LLM_API_KEY_HEADER", "X-API-Key"),
            llm_model=_env("LLM_MODEL", "mistral-medium-2508-ITG"),
            llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 240),
            llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
            llm_max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 6000),
            llm_use_json_mode=_env_bool("LLM_USE_JSON_MODE", False),
            llm_review_model=_env("LLM_REVIEW_MODEL") or None,
            llm_extra_body=_env_json_object("LLM_EXTRA_BODY_JSON"),
            tls_verify=_env_bool("TLS_VERIFY", True),
            ca_bundle_path=_resolve_path(_env("CA_BUNDLE_PATH")),
            enterprise_extra_headers=_env_json_string_object(
                "ENTERPRISE_EXTRA_HEADERS_JSON"
            ),
            http_max_retries=_env_int("HTTP_MAX_RETRIES", 3),
            http_backoff_seconds=_env_float("HTTP_BACKOFF_SECONDS", 2.0),
            vector_store_backend=_env(
                "VECTOR_STORE_BACKEND", "qdrant_local"
            ).lower(),
            qdrant_local_path=qdrant_path,
            qdrant_collection=_env(
                "QDRANT_COLLECTION", "credit_risk_chunks"
            ),
            allow_numpy_fallback=_env_bool("ALLOW_NUMPY_FALLBACK", True),
            save_embeddings_copy=_env_bool("SAVE_EMBEDDINGS_COPY", True),
            retrieval_dense_top_k=_env_int("RETRIEVAL_DENSE_TOP_K", 20),
            retrieval_lexical_top_k=_env_int("RETRIEVAL_LEXICAL_TOP_K", 20),
            retrieval_final_top_k=_env_int("RETRIEVAL_FINAL_TOP_K", 12),
            rrf_k=_env_int("RRF_K", 60),
            max_retrieved_chunks_per_task=_env_int(
                "MAX_RETRIEVED_CHUNKS_PER_TASK", 14
            ),
            min_evidence_fuzzy_score=_env_int(
                "MIN_EVIDENCE_FUZZY_SCORE", 90
            ),
            dedupe_similarity_threshold=_env_int(
                "DEDUPE_SIMILARITY_THRESHOLD", 88
            ),
            report_title=_env(
                "REPORT_TITLE",
                "Rapport des evenements candidats Early Warning Signals",
            ),
            report_confidentiality_label=_env(
                "REPORT_CONFIDENTIALITY_LABEL",
                "CONFIDENTIEL - PROTOTYPE PFE - A VALIDER PAR UN ANALYSTE",
            ),
            run_connectivity_tests=_env_bool("RUN_CONNECTIVITY_TESTS", False),
        )
        instance.validate()
        return instance

    def validate(self) -> None:
        if self.chunk_max_tokens < 100:
            raise ValueError("CHUNK_MAX_TOKENS doit etre >= 100.")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("CHUNK_OVERLAP_TOKENS doit etre >= 0.")
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS doit etre strictement inferieur a CHUNK_MAX_TOKENS."
            )
        if self.embedding_batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE doit etre >= 1.")
        if self.pymupdf_ocr_dpi < 72:
            raise ValueError("PYMUPDF_OCR_DPI doit etre >= 72.")
        if not 0.0 <= self.hybrid_table_overlap_threshold <= 1.0:
            raise ValueError("HYBRID_TABLE_OVERLAP_THRESHOLD doit etre entre 0 et 1.")
        if not 0.0 <= self.picture_description_area_threshold <= 1.0:
            raise ValueError("PICTURE_DESCRIPTION_AREA_THRESHOLD doit etre entre 0 et 1.")
        if self.vector_store_backend not in {"qdrant_local", "numpy"}:
            raise ValueError(
                "VECTOR_STORE_BACKEND doit etre qdrant_local ou numpy."
            )
        valid_auth = {"bearer", "x-api-key", "custom", "none"}
        if self.embedding_auth_mode not in valid_auth:
            raise ValueError("EMBEDDING_AUTH_MODE invalide.")
        if self.llm_auth_mode not in valid_auth:
            raise ValueError("LLM_AUTH_MODE invalide.")

    def ensure_directories(self) -> None:
        for path in (
            self.input_dir,
            self.prepared_dir,
            self.extracted_dir,
            self.chunks_dir,
            self.index_dir,
            self.outputs_dir,
            self.logs_dir,
            self.qdrant_local_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def http_verify(self) -> bool | str:
        if self.ca_bundle_path:
            return str(self.ca_bundle_path)
        return self.tls_verify

    def configuration_summary(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "target_counterparty": self.target_counterparty,
            "embedding_model": self.embedding_model,
            "embedding_base_url_configured": bool(self.embedding_base_url),
            "llm_model": self.llm_model,
            "llm_base_url_configured": bool(self.llm_base_url),
            "vector_store_backend": self.vector_store_backend,
            "qdrant_local_path": str(self.qdrant_local_path),
            "docling_artifacts_path": (
                str(self.docling_artifacts_path)
                if self.docling_artifacts_path
                else None
            ),
            "pymupdf_table_strategy": self.pymupdf_table_strategy,
            "picture_description_enabled": self.picture_description_enabled,
            "picture_description_model": (
                self.picture_description_model or self.llm_model
                if self.picture_description_enabled
                else None
            ),
            "word_to_pdf_required": self.require_word_to_pdf,
        }
