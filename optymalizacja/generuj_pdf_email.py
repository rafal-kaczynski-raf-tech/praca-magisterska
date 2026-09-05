"""Generuje estetyczny raport PDF z trescia wiadomosci e-mail i tabela wynikow
dla prof. Iwanskiego (z pelna obsluga polskich znakow UTF-8).

Uruchomienie:  python -m optymalizacja.generuj_pdf_email
"""
from __future__ import annotations
import os
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


# ---------------------------------------------------------------------------
# Rejestracja czcionek systemowych z pelnym wsparciem UTF-8 (polskie znaki)
# ---------------------------------------------------------------------------
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


class NumberedCanvas(canvas.Canvas):
    """Canvas dodajacy numeracje stron i stopke."""
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
        
        # Stopka
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        
        self.drawString(18 * mm, 10 * mm, "Raport z symulacji sterowania MF-BB | Przetwornica Boost DC-DC")
        self.drawRightString(192 * mm, 10 * mm, f"Strona {self._pageNumber} z {page_count}")
        self.restoreState()


def build_pdf(filename: str = "raport_email_profesor.pdf") -> str:
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        topMargin=14 * mm,
        bottomMargin=18 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    
    # Style niestandardowe z zarejestrowanym fontem UTF-8
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=2,
    )
    
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )

    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=5,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=9,
        leading=13,
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
    story.append(Paragraph("Raport wynikowy: Optymalizacja PSO i scenariusz testowy MF-BB", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=6))

    # Metadane / Nadawca i Odbiorca
    meta_data = [
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Do:</font> Prof. dr hab. inż. Grzegorz Iwański", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Data:</font> Wrzesień 2026", meta_style),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Od:</font> inż. Rafał Kaczyński (album: 342208)", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Temat:</font> Analiza funkcji celu w scenariuszu testowym", meta_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[90 * mm, 84 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 3 * mm))

    # Treść wprowadzenia
    story.append(Paragraph(f"<font name='{FONT_BOLD}'>Szanowny Panie Profesorze,</font>", body_style))
    intro_txt = (
        "Zgodnie z wytycznymi zaimplementowałem pełny scenariusz dynamiczny (T = 0.30 s) "
        "dla przetwornicy podwyższającej DC-DC Boost sterowanej algorytmem Model-Free Bang-Bang (MF-BB). "
        "Scenariusz weryfikuje odporność układu na skoki obciążenia nominalnego "
        "(R_load = 47.619 Ω ↔ 10 MΩ, bieg jałowy) oraz skok napięcia zadanego w dół (240 V → 160 V w t = 0.15 s)."
    )
    story.append(Paragraph(intro_txt, body_style))

    intro_pso = (
        "Dla opisanego scenariusza przeprowadziłem optymalizację parametrów sterownika "
        "(waga prądowa w_i oraz częstotliwość odcięcia filtru f_c_lpf) przy użyciu algorytmu PSO dla 4 funkcji celu. "
        f"Wszystkie wskaźniki jakości wyznaczono w <font name='{FONT_BOLD}'>stanie ustalonym</font> (w ostatnich 40% każdego segmentu czasowego)."
    )
    story.append(Paragraph(intro_pso, body_style))
    story.append(Spacer(1, 1 * mm))

    # Tabela wyników
    story.append(Paragraph("Tabela 1. Wyniki w stanie ustalonym pod obciążeniem (R_load = 47.619 Ω)", h2_style))
    
    headers = [
        Paragraph("Funkcja celu", table_cell_bold),
        Paragraph("Optymalne<br/>(w_i*, f_c*)", table_cell_bold),
        Paragraph("Uchyb v_C<br/>(240 V)", table_cell_bold),
        Paragraph("Tętnienia i_L<br/>p-p (240 V)", table_cell_bold),
        Paragraph("Tętnienia i_L<br/>[%] (240 V)", table_cell_bold),
        Paragraph("Uchyb v_C<br/>(160 V)", table_cell_bold),
        Paragraph("Tętnienia i_L<br/>p-p (160 V)", table_cell_bold),
        Paragraph("Tętnienia i_L<br/>[%] (160 V)", table_cell_bold),
    ]

    rows_data = [
        headers,
        [
            Paragraph(f"<font name='{FONT_BOLD}'>ITAE</font> (napięciowa)", table_cell_left),
            Paragraph("0.2133, 1854 Hz", table_cell),
            Paragraph("-0.29 V", table_cell),
            Paragraph("5.70 A", table_cell),
            Paragraph(f"<font name='{FONT_BOLD}'>47.12%</font>", table_cell),
            Paragraph("+0.17 V", table_cell),
            Paragraph("3.45 A", table_cell),
            Paragraph(f"<font name='{FONT_BOLD}'>64.01%</font>", table_cell),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>CurrentAware</font> (proponowana)", table_cell_left),
            Paragraph("0.2631, 7056 Hz", table_cell),
            Paragraph("-0.35 V", table_cell),
            Paragraph("3.73 A", table_cell),
            Paragraph(f"<font name='{FONT_BOLD}' color='#2B6CB0'>30.83%</font>", table_cell),
            Paragraph("+0.20 V", table_cell),
            Paragraph("2.61 A", table_cell),
            Paragraph(f"<font name='{FONT_BOLD}' color='#2B6CB0'>48.39%</font>", table_cell),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>CurrentOscillation</font>", table_cell_left),
            Paragraph("0.2124, 2039 Hz", table_cell),
            Paragraph("-0.29 V", table_cell),
            Paragraph("6.80 A", table_cell),
            Paragraph("56.24%", table_cell),
            Paragraph("+0.17 V", table_cell),
            Paragraph("3.47 A", table_cell),
            Paragraph("64.45%", table_cell),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>CurrentEffort</font>", table_cell_left),
            Paragraph("0.2131, 2767 Hz", table_cell),
            Paragraph("-0.29 V", table_cell),
            Paragraph("5.70 A", table_cell),
            Paragraph("47.09%", table_cell),
            Paragraph("+0.17 V", table_cell),
            Paragraph("3.45 A", table_cell),
            Paragraph("64.10%", table_cell),
        ],
    ]

    # Strona 1 - glowna tresc raportu i tabela wynikow
    t_res = Table(rows_data, colWidths=[38*mm, 26*mm, 17*mm, 18*mm, 18*mm, 17*mm, 18*mm, 18*mm])
    t_res.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 3.5),
        # Podswietlenie wiersza CurrentAware
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#EBF8FF")),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 8 * mm))

    # Podpis na stronie 1
    sign_txt = (
        f"<font name='{FONT_ITALIC}'>Raport wygenerowany automatycznie z wyników optymalizacji PSO.<br/>"
        f"Z poważaniem,<br/><font name='{FONT_BOLD}'>inż. Rafał Kaczyński</font> (album 342208)</font>"
    )
    story.append(Paragraph(sign_txt, body_style))

    # =========================================================================
    # Strona 2: Załącznik 1 - Przebiegi czasowe w pełnym scenariuszu (0–300 ms)
    # =========================================================================
    plot_pelny = "optymalizacja/wykres_scenariusz_prof_pelny.png"
    if os.path.exists(plot_pelny):
        story.append(PageBreak())
        story.append(Paragraph("Załącznik 1: Przebiegi czasowe w pełnym scenariuszu (0–300 ms)", h2_style))
        story.append(Paragraph(
            "Bezpośrednie przebiegi napięcia $u_{dc}(t)$ oraz prądu cewki $i_L(t)$ dla 4 wariantów funkcji celu "
            "w pełnym horyzoncie czasowym 300 ms. Czerwone linie pionowe oznaczają punkty zdarzeń scenariusza.",
            body_style
        ))
        story.append(Spacer(1, 2 * mm))
        # Zwiększony czytelny wykres na całą wysokość strony 2
        story.append(Image(plot_pelny, width=174 * mm, height=195 * mm))

    # =========================================================================
    # Strona 3: Załącznik 2 - Przybliżenie (Zoom 140–210 ms)
    # =========================================================================
    plot_zoom = "optymalizacja/wykres_scenariusz_prof_zoom.png"
    if os.path.exists(plot_zoom):
        story.append(PageBreak())
        story.append(Paragraph("Załącznik 2: Przybliżenie (Zoom 140–210 ms) – pełna rozdzielczość próbek", h2_style))
        story.append(Paragraph(
            "Przebiegi w oknie 140–210 ms w pełnej rozdzielczości próbek fizyki ($dt = 0.5\\ \\mu\\text{s}$) "
            "obejmujące skok referencji $240 \\to 160\\text{ V}$ w $t=150\\text{ ms}$ oraz wyłączenie urządzenia w $t=200\\text{ ms}$.",
            body_style
        ))
        story.append(Spacer(1, 2 * mm))
        # Zwiększony czytelny wykres na całą wysokość strony 3
        story.append(Image(plot_zoom, width=174 * mm, height=195 * mm))

    doc.build(story, canvasmaker=NumberedCanvas)
    return filename


if __name__ == "__main__":
    out = build_pdf()
    print(f"Wygenerowano PDF: {out}")
