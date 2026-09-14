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
    text = text.strip()
    if not text:
        return []

    return [
        part.strip()
        for part in text_splitter.split_text(text)
        if part.strip()
    ]


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (
        len(lines) >= 3
        and lines[0].startswith("|")
        and lines[1].startswith("|")
        and "---" in lines[1]
    )


def _parse_markdown_row(row: str) -> list[str]:
    row = row.strip()

    if not row.startswith("|"):
        return []

    if row.endswith("|"):
        row = row[1:-1]
    else:
        row = row[1:]

    return [cell.strip() for cell in row.split("|")]


def _build_partial_row(
    cells: list[str],
    start: int,
    end: int,
) -> str:
    partial_cells = [
        cell if start <= index < end else ""
        for index, cell in enumerate(cells)
    ]
    return "| " + " | ".join(partial_cells) + " |"


def _split_text_with_prefix(
    *,
    text: str,
    prefix: str,
    tokenizer: Any,
    max_tokens: int,
) -> list[str]:
    """
    Split text while reserving space for a short semantic prefix.

    This is used for an exceptionally large table cell. We deliberately do
    NOT repeat the full Markdown header here because the header itself can be
    large enough to make every chunk exceed CHUNK_MAX_TOKENS.
    """
    text = text.strip()
    prefix = prefix.strip()

    if not text:
        return []

    prefix_tokens = tokenizer.count_tokens(prefix)

    # Keep a safety margin because token counts are not perfectly additive.
    available_tokens = max_tokens - prefix_tokens - 20

    # If even the prefix is too large, use a minimal prefix.
    if available_tokens < 80:
        prefix = "[TABLE CELL]"
        prefix_tokens = tokenizer.count_tokens(prefix)
        available_tokens = max(80, max_tokens - prefix_tokens - 20)

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RuntimeError(
            "langchain-text-splitters est requis pour le chunking textuel."
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=available_tokens,
        chunk_overlap=min(50, max(0, available_tokens // 8)),
        length_function=tokenizer.count_tokens,
        separators=["\n\n", "\n", ". ", "; ", ": ", " ", ""],
        keep_separator=True,
    )

    raw_parts = [
        part.strip()
        for part in splitter.split_text(text)
        if part.strip()
    ]

    output: list[str] = []

    for part in raw_parts:
        candidate = f"{prefix}\n{part}".strip()

        # Final hard guarantee. If tokenisation non-additivity still pushes the
        # candidate over the limit, reduce it further until it fits.
        if tokenizer.count_tokens(candidate) <= max_tokens:
            output.append(candidate)
            continue

        smaller_budget = max(50, available_tokens - 50)

        while smaller_budget >= 50:
            smaller_splitter = RecursiveCharacterTextSplitter(
                chunk_size=smaller_budget,
                chunk_overlap=0,
                length_function=tokenizer.count_tokens,
                separators=["\n\n", "\n", ". ", "; ", ": ", " ", ""],
                keep_separator=True,
            )

            smaller_parts = [
                p.strip()
                for p in smaller_splitter.split_text(part)
                if p.strip()
            ]

            wrapped = [
                f"{prefix}\n{p}".strip()
                for p in smaller_parts
            ]

            if all(
                tokenizer.count_tokens(chunk) <= max_tokens
                for chunk in wrapped
            ):
                output.extend(wrapped)
                break

            smaller_budget -= 50
        else:
            # Extremely defensive character-level fallback.
            current = ""
            for char in part:
                candidate_char = f"{prefix}\n{current}{char}".strip()

                if (
                    current
                    and tokenizer.count_tokens(candidate_char) > max_tokens
                ):
                    output.append(f"{prefix}\n{current}".strip())
                    current = char
                else:
                    current += char

            if current:
                output.append(f"{prefix}\n{current}".strip())

    return output


def _split_oversized_cell(
    *,
    header: list[str],
    cells: list[str],
    cell_index: int,
    tokenizer: Any,
    max_tokens: int,
) -> list[str]:
    """
    Split one oversized table cell semantically.

    Instead of repeating the whole Markdown table header, preserve only the
    corresponding column name. This prevents wide/verbose headers from making
    every fallback chunk exceed CHUNK_MAX_TOKENS.
    """
    cell_text = cells[cell_index].strip()

    if not cell_text:
        return []

    header_cells = (
        _parse_markdown_row(header[0])
        if header
        else []
    )

    column_name = (
        header_cells[cell_index].strip()
        if cell_index < len(header_cells)
        and header_cells[cell_index].strip()
        else f"column_{cell_index + 1}"
    )

    # Limit a pathological column label so it cannot consume the whole budget.
    if tokenizer.count_tokens(column_name) > 80:
        column_name = f"column_{cell_index + 1}"

    prefix = (
        "[TABLE CELL]\n"
        f"Column: {column_name}\n"
        f"Column index: {cell_index + 1}"
    )

    return _split_text_with_prefix(
        text=cell_text,
        prefix=prefix,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )


def _split_oversized_table_row(
    *,
    header: list[str],
    row: str,
    tokenizer: Any,
    max_tokens: int,
    text_splitter: Any,
) -> list[str]:
    """
    Split an oversized Markdown row.

    Priority:
    1. keep contiguous groups of cells with the full Markdown header;
    2. if one cell alone cannot fit, switch that cell to a compact semantic
       representation: [TABLE CELL] + column name + value;
    3. guarantee that every returned piece is <= max_tokens.
    """
    cells = _parse_markdown_row(row)

    if not cells:
        return _split_text(
            "\n".join([*header, row]),
            text_splitter,
        )

    output: list[str] = []
    group_start = 0

    while group_start < len(cells):
        group_end = group_start
        best_chunk: str | None = None

        while group_end < len(cells):
            candidate_row = _build_partial_row(
                cells=cells,
                start=group_start,
                end=group_end + 1,
            )

            candidate = "\n".join(
                [*header, candidate_row]
            ).strip()

            if tokenizer.count_tokens(candidate) <= max_tokens:
                best_chunk = candidate
                group_end += 1
                continue

            break

        if best_chunk is not None:
            output.append(best_chunk)
            group_start = group_end
            continue

        # This is now an expected handled case, not a warning.
        LOGGER.info(
            "Cellule de tableau trop longue: passage au split semantique "
            "de la cellule | colonne=%s | max_tokens=%s",
            group_start + 1,
            max_tokens,
        )

        output.extend(
            _split_oversized_cell(
                header=header,
                cells=cells,
                cell_index=group_start,
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
        )

        group_start += 1

    # Hard post-condition for this fallback.
    final_output: list[str] = []

    for part in output:
        if tokenizer.count_tokens(part) <= max_tokens:
            final_output.append(part)
        else:
            LOGGER.warning(
                "Chunk de table encore > CHUNK_MAX_TOKENS apres traitement; "
                "fallback final RecursiveCharacterTextSplitter."
            )
            final_output.extend(
                _split_text(part, text_splitter)
            )

    return [
        part.strip()
        for part in final_output
        if part.strip()
    ]


def _split_table(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    text_splitter: Any,
) -> list[str]:
    """
    Table chunking strategy:
    - small table: keep whole;
    - large table: split by rows and repeat Markdown header;
    - oversized row: split by groups of cells;
    - oversized cell: compact semantic cell representation.
    """
    text = text.strip()

    if not text:
        return []

    if tokenizer.count_tokens(text) <= max_tokens:
        return [text]

    if not _looks_like_markdown_table(text):
        return _split_text(
            text,
            text_splitter,
        )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    header = lines[:2]
    rows = lines[2:]

    output: list[str] = []
    current_rows: list[str] = []

    for row in rows:
        candidate = "\n".join(
            [*header, *current_rows, row]
        )

        if (
            current_rows
            and tokenizer.count_tokens(candidate) > max_tokens
        ):
            output.append(
                "\n".join(
                    [*header, *current_rows]
                )
            )
            current_rows = [row]
        else:
            current_rows.append(row)

        single_row = "\n".join(
            [*header, *current_rows]
        )

        if (
            len(current_rows) == 1
            and tokenizer.count_tokens(single_row) > max_tokens
        ):
            LOGGER.info(
                "Ligne de tableau trop longue | tokens=%s | max=%s | "
                "split structurel par cellules",
                tokenizer.count_tokens(single_row),
                max_tokens,
            )

            output.extend(
                _split_oversized_table_row(
                    header=header,
                    row=current_rows[0],
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
                    text_splitter=text_splitter,
                )
            )

            current_rows = []

    if current_rows:
        output.append(
            "\n".join(
                [*header, *current_rows]
            )
        )

    # Global safety check.
    final_output: list[str] = []

    for part in output:
        if tokenizer.count_tokens(part) <= max_tokens:
            final_output.append(part)
        else:
            LOGGER.warning(
                "Chunk de tableau > CHUNK_MAX_TOKENS apres split; "
                "fallback final recursive."
            )
            final_output.extend(
                _split_text(
                    part,
                    text_splitter,
                )
            )

    return [
        part.strip()
        for part in final_output
        if part.strip()
    ]


def _unit_chunks(
    unit: dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
    text_splitter: Any,
) -> list[str]:
    text = str(
        unit.get("text") or ""
    ).strip()

    if unit.get("content_type") == "table":
        return _split_table(
            text=text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            text_splitter=text_splitter,
        )

    return _split_text(
        text,
        text_splitter,
    )


def build_chunks(
    settings: Settings,
) -> list[dict[str, Any]]:
    settings.ensure_directories()

    manifest_path = (
        settings.extracted_dir
        / "extraction_manifest.jsonl"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Le manifest hybride est absent. "
            "Executez scripts/02_extract_with_docling.py."
        )

    tokenizer = _build_tokenizer(
        settings
    )

    text_splitter = _build_recursive_splitter(
        tokenizer=tokenizer,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    accepted_statuses = {
        "success",
        "skipped_existing",
    }

    documents = [
        row
        for row in read_jsonl(
            manifest_path
        )
        if row.get(
            "extraction_status"
        )
        in accepted_statuses
    ]

    if settings.target_counterparty:
        documents = [
            row
            for row in documents
            if row.get(
                "counterparty"
            )
            == settings.target_counterparty
        ]

    all_chunks: list[
        dict[str, Any]
    ] = []

    for manifest in documents:
        units_path = Path(
            str(
                manifest.get(
                    "canonical_units"
                )
                or ""
            )
        )

        if not units_path.exists():
            LOGGER.warning(
                "Canonical units absentes: %s",
                units_path,
            )
            continue

        document_chunk_index = 0

        units = list(
            read_jsonl(
                units_path
            )
        )

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

                page_no = unit.get(
                    "page_number"
                )

                pages = (
                    [int(page_no)]
                    if isinstance(
                        page_no,
                        int,
                    )
                    else []
                )

                content_type = str(
                    unit.get(
                        "content_type"
                    )
                    or "text"
                )

                source_engine = str(
                    unit.get(
                        "source_engine"
                    )
                    or "unknown"
                )

                chunk_id = stable_id(
                    manifest[
                        "document_id"
                    ],
                    unit.get(
                        "unit_id"
                    ),
                    local_index,
                    text,
                    length=32,
                )

                labels = [
                    content_type,
                    source_engine,
                ]

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
            manifest[
                "source_file"
            ],
            document_chunk_index,
        )

    chunks_path = (
        settings.chunks_dir
        / "chunks.jsonl"
    )

    write_jsonl(
        chunks_path,
        all_chunks,
    )

    return all_chunks
