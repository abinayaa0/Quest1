"""
Build Complete System Documentation PDF
Compiles all .md design and documentation files into a single, beautifully formatted PDF:
Video_Dialogue_Localization_Documentation.pdf
"""

import os
import re
from pathlib import Path

import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# List of Markdown documentation files to include in logical order
DOC_FILES = [
    ("1. Project Overview & Quickstart", Path("README.md")),
    ("2. V2 Optimization & Benchmarks", Path("docs/optimization_and_benchmarks_v2.md")),
    ("3. V1 System Documentation", Path("docs/system_documentation_v1.md")),
    ("4. Core Architecture Design", Path("design.md")),
    ("5. Dialogue Matching Design", Path("matching_design.md")),
    ("6. Frame Extraction Design", Path("frame_extraction_design.md")),
]


class NumberedCanvas(canvas.Canvas):
    """Canvas for adding page numbers and running headers to all pages except cover."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Skip cover page

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Header
        self.drawString(54, 11 * inch - 36, "VIDEO DIALOGUE LOCALIZATION SYSTEM — SYSTEM DOCUMENTATION")
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer
        self.setFont("Helvetica", 9)
        self.drawString(54, 36, "Confidential — System Architecture & Design")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.line(54, 48, 8.5 * inch - 54, 48)

        self.restoreState()


def clean_html_for_reportlab(html_str: str) -> str:
    """Sanitize HTML so ReportLab Paragraph parser doesn't crash."""
    # Convert code blocks
    html_str = re.sub(
        r"<code>(.*?)</code>",
        r"<font face='Courier' color='#2B6CB0'><b>\1</b></font>",
        html_str,
        flags=re.DOTALL,
    )
    # Remove unsupported HTML tags
    html_str = re.sub(r"</?(div|span|section|article|header|footer)[^>]*>", "", html_str)
    return html_str.strip()


def parse_markdown_to_flowables(text: str, styles):
    """Convert Markdown text to ReportLab Flowables safely."""
    flowables = []
    lines = text.splitlines()

    in_code = False
    code_lines = []

    for line in lines:
        raw_line = line.rstrip()

        if raw_line.startswith("```"):
            if in_code:
                code_text = "<br/>".join(
                    [
                        c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace(" ", "&nbsp;")
                        for c in code_lines
                    ]
                )
                if code_text.strip():
                    flowables.append(Paragraph(code_text, styles["CodeBlock"]))
                    flowables.append(Spacer(1, 4))
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            continue

        if in_code:
            code_lines.append(raw_line)
            continue

        if not raw_line.strip():
            flowables.append(Spacer(1, 4))
            continue

        # Convert line using python-markdown for safe HTML tag generation
        html_line = markdown.markdown(raw_line)
        cleaned_html = clean_html_for_reportlab(html_line)

        # Map Markdown element types
        if raw_line.startswith("# "):
            clean_text = re.sub(r"<[^>]+>", "", cleaned_html)
            flowables.append(Spacer(1, 10))
            flowables.append(Paragraph(clean_text, styles["Heading1_Custom"]))
            flowables.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3182CE"), spaceAfter=6))
        elif raw_line.startswith("## "):
            clean_text = re.sub(r"<[^>]+>", "", cleaned_html)
            flowables.append(Spacer(1, 8))
            flowables.append(Paragraph(clean_text, styles["Heading2_Custom"]))
            flowables.append(Spacer(1, 3))
        elif raw_line.startswith("### "):
            clean_text = re.sub(r"<[^>]+>", "", cleaned_html)
            flowables.append(Spacer(1, 6))
            flowables.append(Paragraph(clean_text, styles["Heading3_Custom"]))
            flowables.append(Spacer(1, 2))
        elif raw_line.startswith("- ") or raw_line.startswith("* ") or re.match(r"^\d+\.\s", raw_line):
            # Bullet line
            inner = re.sub(r"^<p>(.*?)</p>$", r"\1", cleaned_html, flags=re.DOTALL)
            flowables.append(Paragraph(f"• {inner}", styles["BulletText"]))
        elif raw_line.startswith("---") or raw_line.startswith("***"):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=4))
        else:
            inner = re.sub(r"^<p>(.*?)</p>$", r"\1", cleaned_html, flags=re.DOTALL)
            try:
                flowables.append(Paragraph(inner, styles["BodyCustom"]))
            except Exception:
                # Fallback to plain text if HTML parsing fails
                clean_text = re.sub(r"<[^>]+>", "", raw_line)
                flowables.append(Paragraph(clean_text, styles["BodyCustom"]))

    return flowables


def build_pdf(output_filename="Video_Dialogue_Localization_Documentation.pdf"):
    pdf_path = Path(output_filename).resolve()
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            "CoverTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=26,
            leading=32,
            textColor=colors.HexColor("#1A202C"),
            alignment=1,
            spaceAfter=15,
        )
    )

    styles.add(
        ParagraphStyle(
            "CoverSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#4A5568"),
            alignment=1,
            spaceAfter=25,
        )
    )

    styles.add(
        ParagraphStyle(
            "Heading1_Custom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#2B6CB0"),
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            "Heading2_Custom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#2D3748"),
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            "Heading3_Custom",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=3,
        )
    )

    styles.add(
        ParagraphStyle(
            "BodyCustom",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#2D3748"),
            spaceAfter=4,
        )
    )

    styles.add(
        ParagraphStyle(
            "BulletText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#2D3748"),
            leftIndent=15,
            spaceAfter=2,
        )
    )

    styles.add(
        ParagraphStyle(
            "CodeBlock",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#1A202C"),
            backColor=colors.HexColor("#F7FAFC"),
            borderColor=colors.HexColor("#CBD5E0"),
            borderWidth=0.8,
            borderPadding=6,
            spaceAfter=6,
        )
    )

    story = []

    # Cover Page
    story.append(Spacer(1, 80))
    story.append(Paragraph("🎬 VIDEO DIALOGUE LOCALIZATION SYSTEM", styles["CoverTitle"]))
    story.append(
        Paragraph(
            "Combined Technical Documentation, System Design & V2 Optimization Benchmarks",
            styles["CoverSubtitle"],
        )
    )
    story.append(HRFlowable(width="60%", thickness=2, color=colors.HexColor("#3182CE"), spaceAfter=25))

    # Meta Table
    meta_data = [
        [Paragraph("<b>Document Version:</b>", styles["BodyCustom"]), Paragraph("2.0 (Production)", styles["BodyCustom"])],
        [Paragraph("<b>Pipeline Engine:</b>", styles["BodyCustom"]), Paragraph("V2 Coarse-to-Fine ASR (base/tiny -> small)", styles["BodyCustom"])],
        [Paragraph("<b>Web UI Framework:</b>", styles["BodyCustom"]), Paragraph("Streamlit Web Dashboard", styles["BodyCustom"])],
        [Paragraph("<b>Output Document:</b>", styles["BodyCustom"]), Paragraph("Video_Dialogue_Localization_Documentation.pdf", styles["BodyCustom"])],
    ]
    t_meta = Table(meta_data, colWidths=[130, 260])
    t_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
                ("PADDING", (0, 0), (-1, -1), 8),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(t_meta)
    story.append(Spacer(1, 35))

    # Table of Contents Box
    story.append(Paragraph("<b>Included Documentation Sections:</b>", styles["Heading3_Custom"]))
    toc_data = []
    for title, filepath in DOC_FILES:
        toc_data.append([Paragraph(f"<b>{title}</b>", styles["BodyCustom"]), Paragraph(f"<code>{filepath}</code>", styles["BodyCustom"])])

    t_toc = Table(toc_data, colWidths=[230, 210])
    t_toc.setStyle(
        TableStyle(
            [
                ("PADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(t_toc)
    story.append(PageBreak())

    # Process each Markdown File
    for title, filepath in DOC_FILES:
        if not filepath.exists():
            print(f"Skipping missing file: {filepath}")
            continue

        print(f"Adding documentation section: {title} ({filepath})")
        story.append(Paragraph(f"SECTION: {title.upper()}", styles["Heading1_Custom"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        flowables = parse_markdown_to_flowables(content, styles)
        story.extend(flowables)
        story.append(Spacer(1, 15))
        story.append(PageBreak())

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully compiled PDF documentation to: {pdf_path}")

    # Also save a copy inside output/
    out_copy = Path("output/Video_Dialogue_Localization_Documentation.pdf")
    out_copy.parent.mkdir(parents=True, exist_ok=True)
    with open(pdf_path, "rb") as f_src:
        with open(out_copy, "wb") as f_dst:
            f_dst.write(f_src.read())

    print(f"Saved copy to output directory: {out_copy}")
    return pdf_path


if __name__ == "__main__":
    build_pdf()
