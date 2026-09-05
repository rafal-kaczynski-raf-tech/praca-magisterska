"""Generuje PDF z surowymi probkami t/v_C/i_L (okno stanu ustalonego i skoku)
dla scenariusza prof. Iwanskiego (4 warianty) z pelna obsluga polskich znakow UTF-8.

Uruchomienie:  python -m optymalizacja.generuj_pdf_surowe_probki
"""
from __future__ import annotations
import os
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from optymalizacja.analiza_scenariusz_prof import simulate, CASES

# Czcionki z obsluga polskich znakow
FONT_REGULAR_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

if os.path.exists(FONT_REGULAR_PATH):
    pdfmetrics.registerFont(TTFont("CustomArial", FONT_REGULAR_PATH))
    pdfmetrics.registerFont(TTFont("CustomArial-Bold", FONT_BOLD_PATH))
    FONT_NORMAL = "CustomArial"
    FONT_BOLD = "CustomArial-Bold"
else:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"

# Okna czasowe do wygenerowania surowych probek w stanie ustalonym i w przejsciu
# Okno A: Stan ustalony pod obciazeniem (240V / 47.6 Ohm) w segmencie [0.10, 0.15]s -> probki 145.0 - 147.0 ms (2.0 ms)
# Krok probkowania tabeli: co 20 us (co 2 decyzje sterownika, 100 wierszy)
WINDOWS = [
    {
        "name": "Stan ustalony pod pełnym obciążeniem (240 V, R_load = 47.619 Ω)",
        "t_start": 0.1450,
        "t_end": 0.1470,
        "step": 20e-6,
    },
    {
        "name": "Stan przejściowy: Skok referencji 240 V → 160 V (od t = 150.0 ms)",
        "t_start": 0.1495,
        "t_end": 0.1525,
        "step": 25e-6,
    },
]

OUT_PATH = "surowe_probki_scenariusz_profesor.pdf"


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
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont(FONT_NORMAL, 8)
        self.setFillColor(colors.HexColor("#718096"))
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(15 * mm, 12 * mm, 195 * mm, 12 * mm)
        self.drawString(15 * mm, 8 * mm, "Surowe próbki przebiegów czasowych (t, v_C, i_L, s) | Scenariusz Prof. Iwańskiego")
        self.drawRightString(195 * mm, 8 * mm, f"Strona {self._pageNumber} z {page_count}")
        self.restoreState()


def build_table_data(res: dict, t_start: float, t_end: float, step: float) -> list[list[str]]:
    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]
    s = res["s"]

    # Generujemy wiersze tabeli z rozbiciem na kolumny wielokolumnowe (zeby zmiescic 2 x wiecej danych na stronie)
    raw_points = []
    curr_t = t_start
    while curr_t <= t_end + 1e-12:
        idx = int(np.argmin(np.abs(t - curr_t)))
        raw_points.append((t[idx] * 1e3, v[idx], iL[idx], int(s[idx])))
        curr_t += step

    # Dzielimy na 2 pary kolumn (Lewa i Prawa strona tabeli)
    half = (len(raw_points) + 1) // 2
    table_rows = [[
        "t [ms]", "v_C [V]", "i_L [A]", "s",
        "  |  ",
        "t [ms]", "v_C [V]", "i_L [A]", "s"
    ]]

    for i in range(half):
        p1 = raw_points[i]
        p2 = raw_points[i + half] if (i + half) < len(raw_points) else None
        
        row = [
            f"{p1[0]:.3f}", f"{p1[1]:.2f}", f"{p1[2]:.3f}", f"{p1[3]}",
            " | ",
        ]
        if p2:
            row.extend([f"{p2[0]:.3f}", f"{p2[1]:.2f}", f"{p2[2]:.3f}", f"{p2[3]}"])
        else:
            row.extend(["-", "-", "-", "-"])
        table_rows.append(row)

    return table_rows


def generate_pdf():
    doc = SimpleDocTemplate(
        OUT_PATH,
        pagesize=A4,
        topMargin=12 * mm,
        bottomMargin=16 * mm,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=2,
    )

    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=3,
        spaceAfter=3,
    )

    desc_style = ParagraphStyle(
        "DescStyle",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=3,
    )

    story = []

    story.append(Paragraph("Tabela surowych próbek czasowych napięcia i prądu", title_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=4))
    story.append(Paragraph(
        "Zestawienie dyskretnych wartości chwilowych prądu cewki $i_L(t)$, napięcia kondensatora $v_C(t)$ oraz stanu klucza $s(t)$ "
        "dla 4 wariantów funkcji celu w scenariuszu testowym prof. Iwańskiego (krok tabeli $\\Delta t = 20\\ \\mu\\text{s}$).",
        desc_style
    ))
    story.append(Spacer(1, 2 * mm))

    # Wczytanie wynikow
    results = {}
    for name, path in CASES.items():
        if os.path.exists(path):
            d = np.load(path)
            wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
            results[name] = (wi, fc, simulate(wi, fc))

    for w_idx, win in enumerate(WINDOWS):
        for name, (wi, fc, res) in results.items():
            header_text = f"{name}  (w_i* = {wi:.4f}, f_c* = {fc:.0f} Hz) — {win['name']}"
            story.append(Paragraph(header_text, h2_style))
            story.append(Paragraph(
                f"Przedział: {win['t_start']*1e3:.2f} – {win['t_end']*1e3:.2f} ms (krok: {win['step']*1e6:.0f} µs)",
                desc_style
            ))

            rows = build_table_data(res, win["t_start"], win["t_end"], win["step"])
            
            t = Table(
                rows,
                colWidths=[18*mm, 18*mm, 18*mm, 10*mm, 8*mm, 18*mm, 18*mm, 18*mm, 10*mm],
                repeatRows=1,
            )
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                ("FONTNAME", (0, 0), (-1, -1), FONT_NORMAL),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (3, -1), 0.3, colors.HexColor("#CBD5E0")),
                ("GRID", (5, 0), (8, -1), 0.3, colors.HexColor("#CBD5E0")),
                ("LINEBELOW", (0, 0), (-1, 0), 1.0, colors.HexColor("#2B6CB0")),
                ("PADDING", (0, 0), (-1, -1), 1.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]))
            story.append(t)
            story.append(PageBreak())

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Wygenerowano surowe probki: {OUT_PATH}")


if __name__ == "__main__":
    generate_pdf()
