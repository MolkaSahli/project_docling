from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any
from pfe_ews.config import Settings
from pfe_ews.io_utils import read_jsonl, slugify, stable_id, write_jsonl

LOGGER = logging.getLogger(__name__)

SKIP_PYMUPDF_CLASSES = {"page-header", "page-footer", "picture"}
TEXT_CLASSES = {
    "text",
    "title",
    "section-header",
    "caption",
    "list-item",
    "footnote",
    "formula",
}


def _bbox_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return [float(item) for item in value]
    attrs = [getattr(value, name, None) for name in ("l", "t", "r", "b")]
    if all(item is not None for item in attrs):
        return [float(item) for item in attrs]
    return None


def _bbox_area(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _intersection_area(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _overlap_ratio(a: list[float] | None, b: list[float] | None) -> float:
    """Fraction de la plus petite zone couverte par l'intersection."""
    intersection = _intersection_area(a, b)
    denominator = min(_bbox_area(a), _bbox_area(b))
    if denominator <= 0:
        return 0.0
    return intersection / denominator


def _docling_bbox_top_left(document: Any, page_no: int, bbox: Any) -> list[float] | None:
    """Convertit la bbox Docling en coordonnees top-left comme PyMuPDF."""
    pages = getattr(document, "pages", None)
    page = pages.get(page_no) if isinstance(pages, dict) else None
    if page is not None and hasattr(bbox, "to_top_left_origin"):
        page_size = getattr(page, "size", None)
        page_height = getattr(page_size, "height", None)
        if page_height is not None:
            try:
                bbox = bbox.to_top_left_origin(page_height=float(page_height))
            except Exception:
                pass
    return _bbox_list(bbox)


def _build_docling_converter(settings: Settings):
    """Docling = detecteur/reconstructeur de tables + detecteur/descripteur d'images."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            EasyOcrOptions,
            PdfPipelineOptions,
            TableFormerMode,
            smolvlm_picture_description,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling n'est pas installe. Installez requirements.txt depuis le miroir interne."
        ) from exc

    kwargs: dict[str, Any] = {}
    if settings.docling_artifacts_path:
        kwargs["artifacts_path"] = settings.docling_artifacts_path

    try:
        options = PdfPipelineOptions(**kwargs)
    except TypeError:
        options = PdfPipelineOptions()
        if settings.docling_artifacts_path and hasattr(options, "artifacts_path"):
            options.artifacts_path = settings.docling_artifacts_path

    # OCR et tables Docling.
    options.do_ocr = settings.docling_ocr_enabled
    options.do_table_structure = True
    try:
        options.ocr_options = EasyOcrOptions(
            lang=list(settings.docling_ocr_languages),
            force_full_page_ocr=settings.docling_force_full_page_ocr,
        )
    except TypeError:
        options.ocr_options = EasyOcrOptions(lang=list(settings.docling_ocr_languages))
        if hasattr(options.ocr_options, "force_full_page_ocr"):
            options.ocr_options.force_full_page_ocr = (
                settings.docling_force_full_page_ocr
            )

    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = settings.docling_do_cell_matching

    # Description locale des PictureItem avec le preset SmolVLM de Docling.
    # Aucun endpoint LLM / aucune API key n'est utilise pour cette etape.
    if settings.picture_description_enabled:
        options.generate_picture_images = True
        options.images_scale = settings.picture_description_images_scale
        options.do_picture_description = True

        # Important en environnement entreprise / offline : ne pas autoriser
        # Docling a appeler un service distant pour la description d'images.
        if hasattr(options, "enable_remote_services"):
            options.enable_remote_services = False

        # Le preset pointe vers HuggingFaceTB/SmolVLM-256M-Instruct.
        # Avec DOCLING_ARTIFACTS_PATH renseigne, Docling le charge depuis
        # le dossier local d'artifacts. deepcopy evite de modifier le preset
        # global de Docling pour les conversions suivantes.
        picture_options = copy.deepcopy(smolvlm_picture_description)
        picture_options.prompt = settings.picture_description_prompt
        picture_options.picture_area_threshold = (
            settings.picture_description_area_threshold
        )

        # Compatibilite avec les versions de Docling exposant generation_config.
        if hasattr(picture_options, "generation_config"):
            generation_config = dict(
                getattr(picture_options, "generation_config", {}) or {}
            )
            generation_config["max_new_tokens"] = (
                settings.picture_description_max_tokens
            )
            generation_config["do_sample"] = False
            picture_options.generation_config = generation_config

        # Certaines versions recentes exposent aussi un scale sur les options VLM.
        if hasattr(picture_options, "scale"):
            picture_options.scale = settings.picture_description_images_scale

        options.picture_description_options = picture_options
    elif hasattr(options, "enable_remote_services"):
        options.enable_remote_services = False

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _extract_pymupdf_units(pdf_path: Path, settings: Settings) -> list[dict[str, Any]]:
    """PyMuPDF4LLM = source principale pour texte + tables non couvertes par Docling."""
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError(
            "pymupdf4llm est requis. Ajoutez-le dans requirements.txt / miroir interne."
        ) from exc

    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        force_text=True,
        use_ocr=True,
        ocr_language=settings.pymupdf_ocr_language,
        ocr_dpi=settings.pymupdf_ocr_dpi,
        table_strategy=settings.pymupdf_table_strategy,
        header=False,
        footer=False,
        show_progress=False,
    )
    if not isinstance(pages, list):
        raise RuntimeError("PyMuPDF4LLM n'a pas retourne de page_chunks.")

    units: list[dict[str, Any]] = []
    for page in pages:
        metadata = page.get("metadata") or {}
        page_no = int(metadata.get("page_number") or 0)
        page_text = str(page.get("text") or "")
        boxes = page.get("page_boxes") or []

        if not boxes:
            if page_text.strip():
                units.append(
                    {
                        "page_number": page_no or None,
                        "bbox": None,
                        "content_type": "text",
                        "source_engine": "pymupdf4llm_page_fallback",
                        "text": page_text.strip(),
                        "metadata": {"layout_class": "page_fallback"},
                    }
                )
            continue

        for box in boxes:
            box_class = str(box.get("class") or "text").lower()
            if box_class in SKIP_PYMUPDF_CLASSES:
                # Les images sont gerees par Docling PictureItem + picture description.
                continue

            bbox = _bbox_list(box.get("bbox"))
            pos = box.get("pos") or [0, 0]
            try:
                start, stop = int(pos[0]), int(pos[1])
                text = page_text[start:stop].strip()
            except Exception:
                text = ""
            if not text:
                continue

            if box_class == "table":
                units.append(
                    {
                        "page_number": page_no or None,
                        "bbox": bbox,
                        "content_type": "table",
                        "source_engine": "pymupdf4llm_table_fallback",
                        "text": text,
                        "metadata": {"layout_class": box_class},
                    }
                )
            else:
                content_type = box_class if box_class in TEXT_CLASSES else "text"
                units.append(
                    {
                        "page_number": page_no or None,
                        "bbox": bbox,
                        "content_type": content_type,
                        "source_engine": "pymupdf4llm",
                        "text": text,
                        "metadata": {"layout_class": box_class},
                    }
                )
    return units


def _picture_description_text(picture: Any) -> str:
    """Compatibilite avec les variantes recentes/anciennes de docling-core."""
    meta = getattr(picture, "meta", None)
    description = getattr(meta, "description", None) if meta is not None else None
    text = getattr(description, "text", None)
    if text:
        return str(text).strip()

    for annotation in getattr(picture, "annotations", []) or []:
        if annotation.__class__.__name__ == "PictureDescriptionData":
            value = getattr(annotation, "text", None)
            if value:
                return str(value).strip()
    return ""


def _extract_docling_units(
    document: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extrait toutes les tables detectees par Docling + descriptions des images Docling."""
    tables: list[dict[str, Any]] = []
    pictures: list[dict[str, Any]] = []

    for table_index, table in enumerate(getattr(document, "tables", []) or []):
        prov = (getattr(table, "prov", None) or [None])[0]
        page_no = int(getattr(prov, "page_no", 0) or 0)
        bbox = _docling_bbox_top_left(document, page_no, getattr(prov, "bbox", None))
        try:
            markdown = table.export_to_markdown(doc=document).strip()
        except TypeError:
            markdown = table.export_to_markdown().strip()
        if not markdown:
            continue
        tables.append(
            {
                "page_number": page_no or None,
                "bbox": bbox,
                "content_type": "table",
                "source_engine": "docling_tableformer_accurate",
                "text": markdown,
                "metadata": {"table_index": table_index},
            }
        )

    try:
        from docling_core.types.doc import PictureItem
    except ImportError:
        try:
            from docling_core.types.doc.document import PictureItem
        except ImportError:
            PictureItem = None  # type: ignore[assignment,misc]

    if PictureItem is not None:
        picture_index = 0
        for item, _level in document.iterate_items():
            if not isinstance(item, PictureItem):
                continue
            prov = (getattr(item, "prov", None) or [None])[0]
            page_no = int(getattr(prov, "page_no", 0) or 0)
            bbox = _docling_bbox_top_left(document, page_no, getattr(prov, "bbox", None))
            description = _picture_description_text(item)
            if not description:
                # Si picture description est desactivee ou a echoue, on garde la detection
                # dans les stats mais on n'indexe pas une description vide.
                picture_index += 1
                continue
            try:
                caption = str(item.caption_text(document) or "").strip()
            except Exception:
                caption = ""

            parts = ["[IMAGE / FIGURE]"]
            if caption:
                parts.append(f"Caption: {caption}")
            parts.append(f"Description: {description}")
            pictures.append(
                {
                    "page_number": page_no or None,
                    "bbox": bbox,
                    "content_type": "visual",
                    "source_engine": "docling_picture_description",
                    "text": "\n\n".join(parts),
                    "metadata": {
                        "picture_index": picture_index,
                        "caption": caption or None,
                    },
                }
            )
            picture_index += 1

    return tables, pictures


def _merge_units(
    pymupdf_units: list[dict[str, Any]],
    docling_tables: list[dict[str, Any]],
    docling_pictures: list[dict[str, Any]],
    table_overlap_threshold: float,
) -> list[dict[str, Any]]:
    """Priorite: Docling table > PyMuPDF table. Texte PyMuPDF reste principal."""
    tables_by_page: dict[int, list[dict[str, Any]]] = {}
    pictures_by_page: dict[int, list[dict[str, Any]]] = {}
    for item in docling_tables:
        page_no = item.get("page_number")
        if isinstance(page_no, int):
            tables_by_page.setdefault(page_no, []).append(item)
    for item in docling_pictures:
        page_no = item.get("page_number")
        if isinstance(page_no, int):
            pictures_by_page.setdefault(page_no, []).append(item)

    kept: list[dict[str, Any]] = []
    for item in pymupdf_units:
        page_no = item.get("page_number")
        bbox = item.get("bbox")

        # Regle demandee: si Docling detecte la table, sa version remplace PyMuPDF.
        if item.get("content_type") == "table" and isinstance(page_no, int):
            if any(
                _overlap_ratio(bbox, table.get("bbox")) >= table_overlap_threshold
                for table in tables_by_page.get(page_no, [])
            ):
                continue

        # Evite de dupliquer du texte interne a une image deja resumee par le VLM.
        # Les tables ne sont jamais supprimees par cette regle.
        if item.get("content_type") != "table" and isinstance(page_no, int) and bbox:
            if any(
                _overlap_ratio(bbox, picture.get("bbox")) >= 0.80
                for picture in pictures_by_page.get(page_no, [])
            ):
                continue

        kept.append(item)

    merged = [*kept, *docling_tables, *docling_pictures]
    merged.sort(
        key=lambda item: (
            int(item.get("page_number") or 0),
            float((item.get("bbox") or [0, 0, 0, 0])[1]),
            float((item.get("bbox") or [0, 0, 0, 0])[0]),
            str(item.get("content_type") or ""),
        )
    )
    return merged


def _assign_unit_ids(
    units: list[dict[str, Any]],
    document_id: str,
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, item in enumerate(units):
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        unit_id = stable_id(
            document_id,
            item.get("page_number"),
            item.get("content_type"),
            index,
            text,
            length=32,
        )
        output.append({"unit_id": unit_id, **common, **item})
    return output


def _write_merged_markdown(path: Path, units: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    current_page: int | None = None
    for unit in units:
        page_no = unit.get("page_number")
        if isinstance(page_no, int) and page_no != current_page:
            current_page = page_no
            lines.append(f"\n\n## Page {page_no}\n")
        if unit.get("content_type") in {"table", "visual"}:
            lines.append(
                f"\n<!-- {unit.get('content_type')} | {unit.get('source_engine')} -->\n"
            )
        lines.append(str(unit.get("text") or "").strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _extract_pdf(
    *,
    prepared: dict[str, Any],
    prepared_path: Path,
    output_dir: Path,
    base_name: str,
    settings: Settings,
    converter: Any,
) -> dict[str, Any]:
    canonical_path = output_dir / f"{base_name}.canonical.jsonl"
    merged_md_path = output_dir / f"{base_name}.merged.md"
    docling_json_path = output_dir / f"{base_name}.docling.json"

    pymupdf_units = _extract_pymupdf_units(prepared_path, settings)

    result = converter.convert(
        prepared_path,
        raises_on_error=True,
        max_num_pages=settings.docling_max_num_pages,
        max_file_size=settings.docling_max_file_size_mb * 1024 * 1024,
    )
    document = result.document
    docling_json_path.write_text(
        json.dumps(document.export_to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    docling_tables, docling_pictures = _extract_docling_units(document)

    merged = _merge_units(
        pymupdf_units,
        docling_tables,
        docling_pictures,
        table_overlap_threshold=settings.hybrid_table_overlap_threshold,
    )

    common = {
        "document_id": prepared["document_id"],
        "counterparty": prepared["counterparty"],
        "source_file": prepared["source_file"],
        "source_path": prepared["source_path"],
        "prepared_path": prepared["prepared_path"],
        "source_extension": prepared["source_extension"],
        "prepared_extension": prepared["prepared_extension"],
        "document_type": prepared["document_type"],
    }
    units = _assign_unit_ids(merged, str(prepared["document_id"]), common)
    write_jsonl(canonical_path, units)
    _write_merged_markdown(merged_md_path, units)

    return {
        "canonical_units": str(canonical_path.resolve()),
        "merged_markdown": str(merged_md_path.resolve()),
        "docling_json": str(docling_json_path.resolve()),
        "num_units": len(units),
        "num_tables": sum(unit.get("content_type") == "table" for unit in units),
        "num_docling_tables": len(docling_tables),
        "num_pymupdf_tables": sum(
            unit.get("content_type") == "table"
            and str(unit.get("source_engine", "")).startswith("pymupdf4llm")
            for unit in units
        ),
        "num_visuals": len(docling_pictures),
        "num_pages": len(getattr(document, "pages", {}) or {}),
    }


def _extract_non_pdf_with_docling(
    *,
    prepared: dict[str, Any],
    prepared_path: Path,
    output_dir: Path,
    base_name: str,
) -> dict[str, Any]:
    # 01_prepare_documents convertit normalement Word -> PDF. Ce fallback garde
    # la compatibilite avec Excel/CSV ou un Word direct autorise.
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError("Docling est requis pour les formats non PDF.") from exc

    result = DocumentConverter().convert(prepared_path, raises_on_error=True)
    text = result.document.export_to_markdown().strip()
    canonical_path = output_dir / f"{base_name}.canonical.jsonl"
    merged_md_path = output_dir / f"{base_name}.merged.md"
    common = {
        "document_id": prepared["document_id"],
        "counterparty": prepared["counterparty"],
        "source_file": prepared["source_file"],
        "source_path": prepared["source_path"],
        "prepared_path": prepared["prepared_path"],
        "source_extension": prepared["source_extension"],
        "prepared_extension": prepared["prepared_extension"],
        "document_type": prepared["document_type"],
    }
    units = _assign_unit_ids(
        [
            {
                "page_number": None,
                "bbox": None,
                "content_type": "text",
                "source_engine": "docling_non_pdf_fallback",
                "text": text,
                "metadata": {},
            }
        ],
        str(prepared["document_id"]),
        common,
    )
    write_jsonl(canonical_path, units)
    _write_merged_markdown(merged_md_path, units)
    return {
        "canonical_units": str(canonical_path.resolve()),
        "merged_markdown": str(merged_md_path.resolve()),
        "docling_json": None,
        "num_units": len(units),
        "num_tables": 0,
        "num_docling_tables": 0,
        "num_pymupdf_tables": 0,
        "num_visuals": 0,
        "num_pages": 0,
    }


def extract_documents(settings: Settings) -> list[dict[str, Any]]:
    settings.ensure_directories()
    prepared_manifest = settings.prepared_dir / "prepared_manifest.jsonl"
    if not prepared_manifest.exists():
        raise FileNotFoundError(
            "Le manifest prepare est absent. Executez d'abord scripts/01_prepare_documents.py."
        )

    prepared_rows = [
        row for row in read_jsonl(prepared_manifest) if row.get("status") == "success"
    ]
    if settings.target_counterparty:
        prepared_rows = [
            row
            for row in prepared_rows
            if row.get("counterparty") == settings.target_counterparty
        ]
    if not prepared_rows:
        raise ValueError("Aucun document prepare avec succes.")

    # Construire Docling une seule fois pour tous les PDF.
    converter = _build_docling_converter(settings)
    extraction_rows: list[dict[str, Any]] = []

    for prepared in prepared_rows:
        prepared_path = Path(str(prepared["prepared_path"]))
        output_dir = settings.extracted_dir / slugify(str(prepared["counterparty"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"{prepared['document_id']}__{slugify(str(prepared['source_file']))}"
        canonical_path = output_dir / f"{base_name}.canonical.jsonl"
        merged_md_path = output_dir / f"{base_name}.merged.md"

        record: dict[str, Any] = {
            **prepared,
            "canonical_units": str(canonical_path.resolve()),
            "merged_markdown": str(merged_md_path.resolve()),
            "extraction_status": "pending",
            "extraction_seconds": None,
            "error": None,
        }

        if (
            not settings.force_reextract
            and canonical_path.exists()
            and merged_md_path.exists()
        ):
            record["extraction_status"] = "skipped_existing"
            extraction_rows.append(record)
            LOGGER.info("Extraction hybride deja presente: %s", prepared_path.name)
            continue

        started = time.perf_counter()
        try:
            if prepared_path.suffix.lower() == ".pdf":
                details = _extract_pdf(
                    prepared=prepared,
                    prepared_path=prepared_path,
                    output_dir=output_dir,
                    base_name=base_name,
                    settings=settings,
                    converter=converter,
                )
            else:
                details = _extract_non_pdf_with_docling(
                    prepared=prepared,
                    prepared_path=prepared_path,
                    output_dir=output_dir,
                    base_name=base_name,
                )
            record.update(details)
            record["extraction_status"] = "success"
            record["extraction_seconds"] = round(time.perf_counter() - started, 3)
            LOGGER.info(
                "Hybrid extraction: %s | tables_docling=%s | tables_pymupdf=%s | visuals=%s | %.1fs",
                prepared["source_file"],
                record.get("num_docling_tables"),
                record.get("num_pymupdf_tables"),
                record.get("num_visuals"),
                record["extraction_seconds"],
            )
        except Exception as exc:
            LOGGER.exception("Echec extraction hybride: %s", prepared_path)
            record.update(
                {
                    "extraction_status": "error",
                    "extraction_seconds": round(time.perf_counter() - started, 3),
                    "error": str(exc),
                }
            )
        extraction_rows.append(record)

    manifest_path = settings.extracted_dir / "extraction_manifest.jsonl"
    write_jsonl(manifest_path, extraction_rows)
    return extraction_rows
