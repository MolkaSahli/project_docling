from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pfe_ews.chunking import _build_tokenizer
from pfe_ews.config import Settings
from pfe_ews.io_utils import read_jsonl, stable_id, write_jsonl

LOGGER = logging.getLogger(__name__)


def _build_recursive_splitter(
    tokenizer: Any,
    max_tokens: int,
    overlap_tokens: int,
):
    """Build the standard splitter used for non-table canonical units.

    length_function makes chunk_size/chunk_overlap token-based instead of
    character-based, while RecursiveCharacterTextSplitter still prefers
    paragraph/newline/space boundaries.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "langchain-text-splitters est requis pour le chunking textuel. "
            "Ajoutez-le dans requirements.txt / miroir interne."
        ) from exc

    return RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=overlap_tokens,
        length_function=tokenizer.count_tokens,
        separators=["\n\n", "\n", ". ", "; ", ": ", " ", ""],
        keep_separator=True,
    )


def _split_text(text: str, text_splitter: Any) -> list[str]:
    """RecursiveCharacterTextSplitter for text, headings and visual summaries."""
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in text_splitter.split_text(text) if part.strip()]


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (
        len(lines) >= 3
        and lines[0].startswith("|")
        and lines[1].startswith("|")
        and "---" in lines[1]
    )


def _split_table(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    text_splitter: Any,
) -> list[str]:
    """Keep small tables intact; split large Markdown tables by rows.

    The two Markdown header lines are repeated in every produced chunk.
    If a table is not in Markdown pipe-table form, fall back to the standard
    recursive text splitter rather than using a second custom text chunker.
    """
    text = text.strip()
    if not text:
        return []

    if tokenizer.count_tokens(text) <= max_tokens:
        return [text]

    if not _looks_like_markdown_table(text):
        return _split_text(text, text_splitter)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = lines[:2]
    rows = lines[2:]

    output: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate = "\n".join([*header, *current_rows, row])

        if current_rows and tokenizer.count_tokens(candidate) > max_tokens:
            output.append("\n".join([*header, *current_rows]))
            current_rows = [row]
        else:
            current_rows.append(row)

        # Rare fallback: one single row is itself larger than the limit.
        single_row = "\n".join([*header, *current_rows])
        if len(current_rows) == 1 and tokenizer.count_tokens(single_row) > max_tokens:
            LOGGER.warning(
                "Une ligne de tableau depasse CHUNK_MAX_TOKENS; "
                "fallback RecursiveCharacterTextSplitter pour cette ligne."
            )
            output.extend(_split_text(single_row, text_splitter))
            current_rows = []

    if current_rows:
        output.append("\n".join([*header, *current_rows]))

    return [part.strip() for part in output if part.strip()]


def _unit_chunks(
    unit: dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
    text_splitter: Any,
) -> list[str]:
    text = str(unit.get("text") or "").strip()

    if unit.get("content_type") == "table":
        return _split_table(
            text=text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            text_splitter=text_splitter,
        )

    return _split_text(text, text_splitter)


def build_chunks(settings: Settings) -> list[dict[str, Any]]:
    settings.ensure_directories()
    manifest_path = settings.extracted_dir / "extraction_manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Le manifest hybride est absent. Executez scripts/02_extract_with_docling.py."
        )

    tokenizer = _build_tokenizer(settings)
    text_splitter = _build_recursive_splitter(
        tokenizer=tokenizer,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    accepted_statuses = {"success", "skipped_existing"}
    documents = [
        row
        for row in read_jsonl(manifest_path)
        if row.get("extraction_status") in accepted_statuses
    ]
    if settings.target_counterparty:
        documents = [
            row
            for row in documents
            if row.get("counterparty") == settings.target_counterparty
        ]

    all_chunks: list[dict[str, Any]] = []
    for manifest in documents:
        units_path = Path(str(manifest.get("canonical_units") or ""))
        if not units_path.exists():
            LOGGER.warning("Canonical units absentes: %s", units_path)
            continue

        document_chunk_index = 0
        units = list(read_jsonl(units_path))

        for unit in units:
            parts = _unit_chunks(
                unit=unit,
                tokenizer=tokenizer,
                max_tokens=settings.chunk_max_tokens,
                text_splitter=text_splitter,
            )

            for local_index, text in enumerate(parts):
                if not text.strip():
                    continue

                page_no = unit.get("page_number")
                pages = [int(page_no)] if isinstance(page_no, int) else []
                content_type = str(unit.get("content_type") or "text")
                source_engine = str(unit.get("source_engine") or "unknown")

                chunk_id = stable_id(
                    manifest["document_id"],
                    unit.get("unit_id"),
                    local_index,
                    text,
                    length=32,
                )

                labels = [content_type, source_engine]
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": manifest["document_id"],
                        "counterparty": manifest["counterparty"],
                        "source_file": manifest["source_file"],
                        "source_path": manifest["source_path"],
                        "prepared_path": manifest["prepared_path"],
                        "source_extension": manifest["source_extension"],
                        "prepared_extension": manifest["prepared_extension"],
                        "document_type": manifest["document_type"],
                        "chunk_index": document_chunk_index,
                        "text": text.strip(),
                        "raw_text": text.strip(),
                        "page_numbers": pages,
                        "headings": [],
                        "labels": labels,
                        "is_table": content_type == "table",
                        "estimated_token_count": tokenizer.count_tokens(text),
                        "content_type": content_type,
                        "source_engine": source_engine,
                        "bbox": unit.get("bbox"),
                        "unit_id": unit.get("unit_id"),
                        "metadata": unit.get("metadata") or {},
                    }
                )
                document_chunk_index += 1

        LOGGER.info(
            "Canonical chunking: %s -> %d chunks",
            manifest["source_file"],
            document_chunk_index,
        )

    chunks_path = settings.chunks_dir / "chunks.jsonl"
    write_jsonl(chunks_path, all_chunks)
    return all_chunks
