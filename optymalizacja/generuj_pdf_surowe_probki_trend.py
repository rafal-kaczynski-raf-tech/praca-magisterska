"""Generuje PDF z surowymi probkami dyskretnymi t/v_C/i_L/s dla nowego regulatora Trend MF-BB (200 kHz).

Zawiera:
- Okno 1: Stan ustalony pod pelnym obciazeniem (240V, 1.2 kW, krok 20 us)
- Okno 2: Stan przejsciowy skoku 240V -> 160V w t=150ms (krok 20 us)
- Okno 3: Stan ustalony po skoku (160V, 538 W, krok 20 us)

Uruchomienie:  python -m optymalizacja.generuj_pdf_surowe_probki_trend
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

from src.config import ConverterParams, TrendMFBBParams, SimulationConfig
from src.simulator import Simulator
from optymalizacja.scenarios import professor_scenario, PROF_T_END

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

OUT_PATH = "surowe_probki_trend_mfbb.pdf"


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
        self.line(14 * mm, 12 * mm, 196 * mm, 12 * mm)
        self.drawString(14 * mm, 8 * mm, "Surowe próbki dyskretne Trend MF-BB (200 kHz) | Scenariusz Prof. Iwańskiego")
        self.drawRightString(196 * mm, 8 * mm, f"Strona {self._pageNumber} z {page_count}")
        self.restoreState()


def build_table_data(t, v, iL, s, t_start: float, t_end: float, step: float) -> list[list[str]]:
    raw_points = []
    curr_t = t_start
    while curr_t <= t_end + 1e-12:
        idx = int(np.argmin(np.abs(t - curr_t)))
        raw_points.append((t[idx] * 1e3, v[idx], iL[idx], int(s[idx])))
        curr_t += step

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
    conv = ConverterParams(V_in=100.0, L=0.75e-3, C=200e-6, R_load=47.619, v_C0=100.0, i_L0=0.0)
    ctrl = TrendMFBBParams(u_ref=240.0, T_s_ctrl=5e-6, wi=0.3384, wv=0.7194, fc_i=922.9, fc_u=235.1, i_max=20.0, i_min=-20.0)
    cfg = SimulationConfig(converter=conv, controller=ctrl, dt_phys=0.5e-6, T_end=PROF_T_END, scenario=professor_scenario())
    res = Simulator(cfg).run()

    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]
    s = res["s"]

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
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=2,
    )
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=2,
        spaceAfter=2,
    )
    desc_style = ParagraphStyle(
        "DescStyle",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=3,
    )

    story = []

    # Naglowek
    story.append(Paragraph("Tabela surowych próbek dyskretnych – Trend MF-BB ($f_s = 200\\text{ kHz}$, $T_s = 5\\ \\mu\\text{s}$)", title_style))
    story.append(Paragraph(
        "Zestawienie dyskretnych próbek trajektorii czasowych ($t, v_C, i_L, s$) dla nowego regulatora z filtrowanym trendem napięcia "
        "w zadanym scenariuszu dynamicznym. Nastawy optymalne PSO: $w_i^* = 0.3384$, $w_v^* = 0.7194$, $f_{c,i}^* = 922.9\\text{ Hz}$, $f_{c,u}^* = 235.1\\text{ Hz}$.",
        desc_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=3))

    windows_def = [
        ("Okno 1: Stan ustalony pod pełnym obciążeniem (240 V, 1.2 kW, t ∈ [145.0, 146.5] ms, krok 20 µs)", 0.1450, 0.1465, 20e-6),
        ("Okno 2: Stan przejściowy – skok zadanego napięcia 240 V → 160 V (t ∈ [149.8, 151.3] ms, krok 20 µs)", 0.1498, 0.1513, 20e-6),
        ("Okno 3: Stan ustalony po skoku pod obciążeniem (160 V, 538 W, t ∈ [170.0, 171.5] ms, krok 20 µs)", 0.1700, 0.1715, 20e-6),
    ]

    for title, t1, t2, st in windows_def:
        story.append(Paragraph(title, h2_style))
        t_data = build_table_data(t, v, iL, s, t1, t2, st)
        col_w = [18 * mm, 18 * mm, 18 * mm, 10 * mm, 6 * mm, 18 * mm, 18 * mm, 18 * mm, 10 * mm]

        t_elem = Table(t_data, colWidths=col_w, repeatRows=1)
        t_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, 0), 7.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
            ("TOPPADDING", (0, 0), (-1, 0), 2),
            ("FONTNAME", (0, 1), (-1, -1), FONT_NORMAL),
            ("FONTSIZE", (0, 1), (-1, -1), 6.8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 1.0),
            ("TOPPADDING", (0, 1), (-1, -1), 1.0),
            ("GRID", (0, 0), (3, -1), 0.3, colors.HexColor("#CBD5E0")),
            ("GRID", (5, 0), (8, -1), 0.3, colors.HexColor("#CBD5E0")),
            ("BACKGROUND", (4, 0), (4, -1), colors.HexColor("#F7FAFC")),
            ("TEXTCOLOR", (4, 0), (4, -1), colors.HexColor("#A0AEC0")),
        ]
        for r_i in range(1, len(t_data)):
            if r_i % 2 == 0:
                t_style.append(("BACKGROUND", (0, r_i), (3, r_i), colors.HexColor("#F7FAFC")))
                t_style.append(("BACKGROUND", (5, r_i), (8, r_i), colors.HexColor("#F7FAFC")))
        t_elem.setStyle(TableStyle(t_style))
        story.append(t_elem)
        story.append(Spacer(1, 3 * mm))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Wygenerowano PDF z surowymi probkami: {OUT_PATH}")


if __name__ == "__main__":
    generate_pdf()
