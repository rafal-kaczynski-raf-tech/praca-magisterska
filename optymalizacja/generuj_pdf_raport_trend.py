"""Generuje estetyczny, formalny raport PDF z wynikami symulacji dla regulatora
Trend MF-BB (w pelnym scenariuszu dynamicznym 0.30s prof. Iwanskiego)
z pelna obsluga polskich znakow UTF-8.

Struktura 3-stronicowa:
- Strona 1: Metadane, parametry nastaw PSO, obiektywna tabela wynikow dla 5 segmentow, podpis.
- Strona 2: Zalacznik 1 -- pelne przebiegi czasowe (0–300 ms).
- Strona 3: Zalacznik 2 -- zblizenie mikrosekundowe (140–210 ms, probki 0.5 us).

BEZ komentarzy i subiektywnych wnioskow.

Uruchomienie:  python -m optymalizacja.generuj_pdf_raport_trend
"""
from __future__ import annotations
import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONT_REGULAR_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ITALIC_PATH = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
FONT_BOLD_ITALIC_PATH = "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"

if os.path.exists(FONT_REGULAR_PATH):
    pdfmetrics.registerFont(TTFont("CustomArial", FONT_REGULAR_PATH))
    pdfmetrics.registerFont(TTFont("CustomArial-Bold", FONT_BOLD_PATH))
    pdfmetrics.registerFont(TTFont("CustomArial-Italic", FONT_ITALIC_PATH))
    pdfmetrics.registerFont(TTFont("CustomArial-BoldItalic", FONT_BOLD_ITALIC_PATH))
    FONT_NORMAL = "CustomArial"
    FONT_BOLD = "CustomArial-Bold"
    FONT_ITALIC = "CustomArial-Italic"
    FONT_BOLD_ITALIC = "CustomArial-BoldItalic"
else:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"
    FONT_ITALIC = "Helvetica-Oblique"
    FONT_BOLD_ITALIC = "Helvetica-BoldOblique"

OUT_PATH = "raport_trend_mfbb_profesor.pdf"


class NumberedCanvas(canvas.Canvas):
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
        self.saveState()
        self.setFont(FONT_NORMAL, 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        self.drawString(18 * mm, 10 * mm, "Raport z symulacji Trend MF-BB | Przetwornica Boost DC-DC | Politechnika Warszawska")
        self.drawRightString(192 * mm, 10 * mm, f"Strona {self._pageNumber} z {page_count}")
        self.restoreState()


def build_pdf(filename: str = OUT_PATH) -> str:
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=13.5,
        leading=16.5,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=2,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#4A5568"),
    )

    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=10,
        leading=12.5,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=4,
        spaceAfter=2,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8,
        leading=10,
        alignment=1, # Center
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell,
        fontName=FONT_BOLD,
    )

    table_cell_left = ParagraphStyle(
        "TableCellLeft",
        parent=table_cell,
        fontName=FONT_BOLD,
        alignment=0, # Left
    )

    story = []

    # Nagłówek dokumentu
    story.append(Paragraph("Raport wynikowy: Regulator Trend MF-BB z filtrowanym trendem napięcia", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=5))

    # Metadane / Nadawca i Odbiorca
    meta_data = [
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Do:</font> Prof. dr hab. inż. Grzegorz Iwański", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Data:</font> Wrzesień 2026", meta_style),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Od:</font> inż. Rafał Kaczyński (album: 342208)", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Temat:</font> Wyniki symulacji Trend MF-BB (fs = 200 kHz, Ts = 5 us)", meta_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[90 * mm, 84 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("PADDING", (0, 0), (-1, -1), 3.5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 2.5 * mm))

    # Treść wprowadzenia (obiektywna, bez wnioskowania)
    story.append(Paragraph(f"<font name='{FONT_BOLD}'>Szanowny Panie Profesorze,</font>", body_style))
    intro_txt = (
        "W nawiązaniu do noty technicznej przedstawiam zestawienie wyników symulacji dla regulatora "
        "Model-Free Bang-Bang z filtrowanym trendem napięcia $\\Delta u_f$ w zadanym scenariuszu dynamicznym "
        "($T_{\\mathrm{end}} = 0.30\\text{ s}$, $V_{\\mathrm{in}} = 100\\text{ V}$, $L = 750\\ \\mu\\text{H}$, "
        "$C = 200\\ \\mu\\text{F}$, $R_{\\mathrm{load}} = 47.619\\ \\Omega \\leftrightarrow 10\\text{ M}\\Omega$, "
        "skok $240 \\to 160\\text{ V}$, limit $I_{\\max} = 20\\text{ A}$)."
    )
    story.append(Paragraph(intro_txt, body_style))

    pso_txt = (
        "Nastawy regulatora wyznaczone przez algorytm PSO dla funkcji celu CurrentAware:<br/>"
        "• waga błędu prądu: $w_i^* = 0.3384$,<br/>"
        "• waga trendu napięcia: $w_v^* = 0.7194$,<br/>"
        "• filtr punktu pracy prądu: $f_{c,i}^* = 922.9\\text{ Hz}$ ($\\alpha_i = 0.0143$, $\\beta_i = 0.9714$),<br/>"
        "• filtr wolnozmiennego napięcia: $f_{c,u}^* = 235.1\\text{ Hz}$ ($\\alpha_u = 0.0037$, $\\beta_u = 0.9926$).<br/>"
        "Częstotliwość próbkowania sterownika: $f_s = 200\\text{ kHz}$ ($T_s = 5\\ \\mu\\text{s}$), krok całkowania fizyki: $dt_{\\mathrm{phys}} = 0.5\\ \\mu\\text{s}$."
    )
    story.append(Paragraph(pso_txt, body_style))
    story.append(Spacer(1, 1.5 * mm))

    # Tabela 1: Wyniki w stanach ustalonych
    story.append(Paragraph("Tabela 1. Zestawienie wskaźników w stanach ustalonych (ostatnie 40% każdego segmentu)", h2_style))

    headers = [
        Paragraph("Segment czasowy [s]", table_cell_bold),
        Paragraph("U_ref [V]", table_cell_bold),
        Paragraph("Stan obciążenia", table_cell_bold),
        Paragraph("Błąd v_C [V]", table_cell_bold),
        Paragraph("Średni i_L [A]", table_cell_bold),
        Paragraph("Tętnienia i_L<br/>p-p [A]", table_cell_bold),
        Paragraph("Tętnienia i_L<br/>[%]", table_cell_bold),
    ]

    rows_data = [
        headers,
        [
            Paragraph("[0.05, 0.10)", table_cell_left),
            Paragraph("240.0", table_cell),
            Paragraph("OFF (10 MΩ)", table_cell),
            Paragraph("-0.100 V", table_cell),
            Paragraph("-0.047 A", table_cell),
            Paragraph("3.019 A", table_cell),
            Paragraph("N/A (jałowy)", table_cell),
        ],
        [
            Paragraph("[0.10, 0.15)", table_cell_left),
            Paragraph("240.0", table_cell),
            Paragraph("ON (47.62 Ω, 1.2 kW)", table_cell),
            Paragraph("<font name='" + FONT_BOLD + "'>-0.041 V</font>", table_cell),
            Paragraph("12.118 A", table_cell),
            Paragraph("<font name='" + FONT_BOLD + "'>2.971 A</font>", table_cell),
            Paragraph("<font name='" + FONT_BOLD + "'>24.51%</font>", table_cell),
        ],
        [
            Paragraph("[0.15, 0.20)", table_cell_left),
            Paragraph("160.0", table_cell),
            Paragraph("ON (47.62 Ω, 538 W)", table_cell),
            Paragraph("+0.048 V", table_cell),
            Paragraph("5.373 A", table_cell),
            Paragraph("2.131 A", table_cell),
            Paragraph("39.65%", table_cell),
        ],
        [
            Paragraph("[0.20, 0.25)", table_cell_left),
            Paragraph("160.0", table_cell),
            Paragraph("OFF (10 MΩ)", table_cell),
            Paragraph("+0.088 V", table_cell),
            Paragraph("-0.020 A", table_cell),
            Paragraph("2.164 A", table_cell),
            Paragraph("N/A (jałowy)", table_cell),
        ],
        [
            Paragraph("[0.25, 0.30)", table_cell_left),
            Paragraph("160.0", table_cell),
            Paragraph("ON (47.62 Ω, 538 W)", table_cell),
            Paragraph("+0.048 V", table_cell),
            Paragraph("5.374 A", table_cell),
            Paragraph("2.114 A", table_cell),
            Paragraph("39.34%", table_cell),
        ],
    ]

    t_res = Table(rows_data, colWidths=[36 * mm, 20 * mm, 40 * mm, 22 * mm, 20 * mm, 20 * mm, 20 * mm])
    t_res.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 3),
        # Podswietlenie wierszy pod pelnym obciazeniem
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#EBF8FF")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#EBF8FF")),
        ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#EBF8FF")),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 2 * mm))

    peak_txt = "Maksymalny prąd szczytowy zarejestrowany w całym horyzoncie 0–300 ms: $i_{L,\\mathrm{peak}} = 21.31\\text{ A}$ (poszanowanie limitu $I_{\\max} = 20.0\\text{ A}$)."
    story.append(Paragraph(peak_txt, body_style))
    story.append(Spacer(1, 4 * mm))

    # Podpis na stronie 1
    sign_txt = (
        f"<font name='{FONT_ITALIC}'>Raport wygenerowany automatycznie na podstawie symulacji numerycznej.<br/>"
        f"Z poważaniem,<br/><font name='{FONT_BOLD}'>inż. Rafał Kaczyński</font> (album 342208)</font>"
    )
    story.append(Paragraph(sign_txt, body_style))

    # =========================================================================
    # Strona 2: Załącznik 1 - Przebiegi czasowe w pełnym scenariuszu (0–300 ms)
    # =========================================================================
    plot_pelny = "optymalizacja/wykres_trend_mfbb_pelny.png"
    if os.path.exists(plot_pelny):
        story.append(PageBreak())
        story.append(Paragraph("Załącznik 1: Przebiegi czasowe Trend MF-BB w pełnym scenariuszu (0–300 ms)", h2_style))
        story.append(Paragraph(
            "Przebiegi napięcia wyjściowego $u_{dc}(t)$ na tle wartości zadanej $u_{\\mathrm{ref}}(t)$ oraz prądu cewki $i_L(t)$ "
            "w pełnym horyzoncie czasowym 300 ms przy takcie $f_s = 200\\text{ kHz}$. Linie pionowe wskazują momenty komutacji obciążenia i skoku referencji.",
            body_style
        ))
        story.append(Spacer(1, 2 * mm))
        story.append(Image(plot_pelny, width=174 * mm, height=195 * mm))

    # =========================================================================
    # Strona 3: Załącznik 2 - Przybliżenie (Zoom 140–210 ms)
    # =========================================================================
    plot_zoom = "optymalizacja/wykres_trend_mfbb_zoom.png"
    if os.path.exists(plot_zoom):
        story.append(PageBreak())
        story.append(Paragraph("Załącznik 2: Przybliżenie (Zoom 140–210 ms) – pełna rozdzielczość próbek ($dt = 0.5\\ \\mu\\text{s}$)", h2_style))
        story.append(Paragraph(
            "Przebiegi w oknie 140–210 ms w pełnej rozdzielczości próbek silnika fizyki ($dt = 0.5\\ \\mu\\text{s}$) "
            "obejmujące skok zadanego napięcia $240 \\to 160\\text{ V}$ w $t=150\\text{ ms}$ oraz wyłączenie obciążenia w $t=200\\text{ ms}$.",
            body_style
        ))
        story.append(Spacer(1, 2 * mm))
        story.append(Image(plot_zoom, width=174 * mm, height=195 * mm))

    doc.build(story, canvasmaker=NumberedCanvas)
    return filename


if __name__ == "__main__":
    out = build_pdf()
    print(f"Wygenerowano raport PDF: {out}")
