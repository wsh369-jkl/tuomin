from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import pypdfium2 as pdfium

from app.core.runtime_security import ensure_private_directory, ensure_private_file


class PageRenderer:
    def render_pages(
        self,
        *,
        pdf_path: Path,
        page_numbers: Sequence[int],
        output_dir: Path,
        dpi: int = 180,
        max_edge: int = 1900,
    ) -> Dict[int, Path]:
        ensure_private_directory(output_dir)
        result: Dict[int, Path] = {}
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            page_count = len(document)
            for page_number in sorted({int(value) for value in page_numbers if int(value) > 0}):
                if page_number > page_count:
                    continue
                page = document[page_number - 1]
                scale = max(0.5, float(dpi) / 72.0)
                image = page.render(scale=scale).to_pil().convert("RGB")
                image.thumbnail((max_edge, max_edge))
                path = output_dir / f"page_{page_number:04d}.jpg"
                image.save(path, format="JPEG", quality=92)
                ensure_private_file(path)
                result[page_number] = path
        finally:
            document.close()
        return result

    def page_count(self, pdf_path: Path) -> int:
        document = pdfium.PdfDocument(str(pdf_path))
        try:
            return len(document)
        finally:
            document.close()

