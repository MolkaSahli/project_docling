from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

import markdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)

from pfe_ews.config import Settings
from pfe_ews.io_utils import read_jsonl


CSS = """
body {
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f5f6f8;
    color: #202124;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px;
}

.document-header {
    background: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}

.page {
    background: white;
    margin-bottom: 30px;
    padding: 24px;
    border-radius: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
}

.page-title {
    border-bottom: 2px solid #ddd;
    padding-bottom: 8px;
    margin-bottom: 20px;
}

.unit {
    border-left: 4px solid #999;
    padding: 14px 18px;
    margin: 16px 0;
    background: #fafafa;
}

.unit.table {
    border-left-color: #7656a8;
}

.unit.visual {
    border-left-color: #d98324;
}

.unit.title,
.unit.section-header {
    border-left-color: #2878b5;
}

.unit.text {
    border-left-color: #4c956c;
}

.badges {
    margin-bottom: 10px;
}

.badge {
    display: inline-block;
    padding: 3px 8px;
    margin-right: 6px;
    border-radius: 10px;
    background: #e9ecef;
    font-size: 12px;
    font-family: monospace;
}

.metadata {
    font-family: monospace;
    color: #666;
    font-size: 12px;
    margin-bottom: 10px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}

th,
td {
    border: 1px solid #bbb;
    padding: 8px;
    text-align: left;
    vertical-align: top;
}

th {
    background: #eeeeee;
}

.visual-content {
    background: #fff9ef;
    padding: 15px;
    border-radius: 8px;
}
"""


def render_markdown(text: str) -> str:
    return markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
        ],
    )


def render_unit(
    unit: dict[str, Any],
) -> str:

    content_type = str(
        unit.get("content_type") or "text"
    )

    source_engine = str(
        unit.get("source_engine") or "unknown"
    )

    bbox = unit.get("bbox")

    unit_id = unit.get("unit_id")

    raw_text = str(
        unit.get("text") or ""
    )

    content_html = render_markdown(
        raw_text
    )

    if content_type == "visual":
        content_html = (
            '<div class="visual-content">'
            + content_html
            + "</div>"
        )

    return f"""
    <div class="unit {html.escape(content_type)}">

        <div class="badges">

            <span class="badge">
                {html.escape(content_type)}
            </span>

            <span class="badge">
                {html.escape(source_engine)}
            </span>

        </div>

        <div class="metadata">
            unit_id={html.escape(str(unit_id))}
            <br>
            bbox={html.escape(str(bbox))}
        </div>

        {content_html}

    </div>
    """


def generate_html(
    units_path: Path,
    output_path: Path,
) -> None:

    units = list(
        read_jsonl(units_path)
    )

    pages: dict[
        int | None,
        list[dict[str, Any]]
    ] = {}

    for unit in units:

        page_no = unit.get(
            "page_number"
        )

        pages.setdefault(
            page_no,
            [],
        ).append(unit)

    if units:
        source_file = units[0].get(
            "source_file",
            units_path.name,
        )

        counterparty = units[0].get(
            "counterparty",
            "",
        )
    else:
        source_file = units_path.name
        counterparty = ""

    body_parts = []

    for page_no, page_units in pages.items():

        page_label = (
            f"Page {page_no}"
            if page_no is not None
            else "Page inconnue"
        )

        units_html = "\n".join(
            render_unit(unit)
            for unit in page_units
        )

        body_parts.append(
            f"""
            <section class="page">

                <h2 class="page-title">
                    {html.escape(page_label)}
                </h2>

                {units_html}

            </section>
            """
        )

    document_html = f"""
<!DOCTYPE html>

<html lang="fr">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0">

<title>
Extraction - {html.escape(str(source_file))}
</title>

<style>
{CSS}
</style>

</head>

<body>

<div class="container">

    <div class="document-header">

        <h1>
            Extraction documentaire
        </h1>

        <p>
            <strong>Document :</strong>
            {html.escape(str(source_file))}
        </p>

        <p>
            <strong>Contrepartie :</strong>
            {html.escape(str(counterparty))}
        </p>

        <p>
            <strong>Unités extraites :</strong>
            {len(units)}
        </p>

    </div>

    {''.join(body_parts)}

</div>

</body>

</html>
"""

    output_path.write_text(
        document_html,
        encoding="utf-8",
    )


def main() -> None:

    settings = Settings.from_env()

    manifest_path = (
        settings.extracted_dir
        / "extraction_manifest.jsonl"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Executez d'abord le script 02."
        )

    documents = list(
        read_jsonl(manifest_path)
    )

    generated = 0

    for document in documents:

        if document.get(
            "extraction_status"
        ) not in {
            "success",
            "skipped_existing",
        }:
            continue

        canonical = document.get(
            "canonical_units"
        )

        if not canonical:
            continue

        canonical_path = Path(
            str(canonical)
        )

        if not canonical_path.exists():
            continue

        output_path = (
            canonical_path.parent
            / (
                canonical_path.name
                .replace(
                    ".canonical.jsonl",
                    ".extraction.html",
                )
            )
        )

        generate_html(
            canonical_path,
            output_path,
        )

        print(
            f"HTML genere : "
            f"{output_path}"
        )

        generated += 1

    print(
        f"\n{generated} fichier(s) "
        f"HTML genere(s)."
    )


if __name__ == "__main__":
    main()