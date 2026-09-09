from __future__ import annotations

import html
import sys
from pathlib import Path

import markdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pfe_ews.config import Settings


CSS = '''
body {
    font-family: Arial, Helvetica, sans-serif;
    max-width: 1100px;
    margin: 40px auto;
    padding: 0 30px;
    line-height: 1.55;
    color: #222;
    background: white;
}

h1, h2, h3, h4 {
    margin-top: 1.4em;
    margin-bottom: 0.6em;
}

p {
    margin: 0.7em 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    font-size: 14px;
}

th, td {
    border: 1px solid #999;
    padding: 8px 10px;
    text-align: left;
    vertical-align: top;
}

th {
    background: #f2f2f2;
    font-weight: 600;
}

blockquote {
    border-left: 4px solid #bbb;
    margin-left: 0;
    padding-left: 16px;
    color: #555;
}

pre {
    background: #f6f6f6;
    padding: 12px;
    overflow-x: auto;
}

code {
    font-family: Consolas, monospace;
}

hr {
    border: 0;
    border-top: 1px solid #ddd;
    margin: 28px 0;
}
'''


def merged_markdown_to_html(
    markdown_path: Path,
    html_path: Path,
) -> None:
    """Convertit un .merged.md en HTML simple, proche d'un export Docling."""

    md_text = markdown_path.read_text(
        encoding="utf-8",
    )

    rendered = markdown.markdown(
        md_text,
        extensions=[
            "tables",
            "fenced_code",
            "sane_lists",
        ],
    )

    title = markdown_path.name.replace(
        ".merged.md",
        "",
    )

    document = f'''<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{CSS}
</style>
</head>

<body>
{rendered}
</body>
</html>
'''

    html_path.write_text(
        document,
        encoding="utf-8",
    )


def main() -> None:
    settings = Settings.from_env()

    extracted_dir = settings.extracted_dir

    merged_files = sorted(
        extracted_dir.rglob("*.merged.md")
    )

    if not merged_files:
        print(
            "Aucun fichier .merged.md trouve dans :",
            extracted_dir,
        )
        print(
            "Executez d'abord le script 02."
        )
        return

    generated = 0

    for markdown_path in merged_files:

        html_path = markdown_path.with_name(
            markdown_path.name.replace(
                ".merged.md",
                ".extraction.html",
            )
        )

        merged_markdown_to_html(
            markdown_path,
            html_path,
        )

        print(
            f"HTML genere : {html_path}"
        )

        generated += 1

    print()
    print(
        f"{generated} fichier(s) HTML genere(s)."
    )


if __name__ == "__main__":
    main()
