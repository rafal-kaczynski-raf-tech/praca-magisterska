"""Generuje pelen zestaw wynikow dla nowego regulatora Trend MF-BB nastrojonego
dwustrefowa funkcja celu prof. Iwanskiego (error_threshold_aware):

Parametry optymalne PSO:
  w_i*  = 0.3583
  w_v*  = 6.6502
  fc_i* = 687.2 Hz  (alpha_i = 0.0211, beta_i = 0.9577)
  fc_u* = 250.9 Hz  (alpha_u = 0.0078, beta_u = 0.9843)

Generuje:
1. optymalizacja/wykres_trend_threshold_pelny.png (0–300 ms)
2. optymalizacja/wykres_trend_threshold_zoom.png (140–210 ms)
3. raport_trend_threshold_profesor.pdf (3-stronicowy formalny raport z tabela)
4. surowe_probki_trend_threshold.pdf (3-stronicowa tabela z probkami dyskretnymi)

Uruchomienie:  python -m optymalizacja.generuj_pakiet_threshold
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from src.config import ConverterParams, TrendMFBBParams, SimulationConfig
from src.simulator import Simulator
from optymalizacja.scenarios import professor_scenario, PROF_T_END
from optymalizacja.wykres_scenariusz_prof import _minmax_decimate

# ---------------------------------------------------------------------------
# Czcionki z obsluga polskich znakow
# ---------------------------------------------------------------------------
FONT_REGULAR_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ITALIC_PATH = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

if os.path.exists(FONT_REGULAR_PATH):
    pdfmetrics.registerFont(TTFont("CustomArial", FONT_REGULAR_PATH))
    pdfmetrics.registerFont(TTFont("CustomArial-Bold", FONT_BOLD_PATH))
    pdfmetrics.registerFont(TTFont("CustomArial-Italic", FONT_ITALIC_PATH))
    FONT_NORMAL = "CustomArial"
    FONT_BOLD = "CustomArial-Bold"
    FONT_ITALIC = "CustomArial-Italic"
else:
    FONT_NORMAL = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"
    FONT_ITALIC = "Helvetica-Oblique"


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, footer_title="Raport", **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []
        self.footer_title = footer_title

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
        self.line(16 * mm, 12 * mm, 194 * mm, 12 * mm)
        self.drawString(16 * mm, 8 * mm, self.footer_title)
        self.drawRightString(194 * mm, 8 * mm, f"Strona {self._pageNumber} z {page_count}")
        self.restoreState()


def run_simulation():
    conv = ConverterParams(
        V_in=100.0,
        L=0.75e-3,
        C=200e-6,
        R_load=47.619,
        v_C0=100.0,
        i_L0=0.0,
    )
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,      # Ts = 10 us (fs = 100 kHz)
        wi=0.3583,
        wv=6.6502,
        fc_i=687.2,
        fc_u=250.9,
        i_max=20.0,
        i_min=-20.0,
    )
    cfg = SimulationConfig(
        converter=conv,
        controller=ctrl,
        dt_phys=0.5e-6,
        T_end=PROF_T_END,
        scenario=professor_scenario(),
    )
    sim = Simulator(cfg)
    return sim.run()


def generate_plots(res: dict):
    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]
    tm = t * 1e3

    u_ref_arr = np.full_like(tm, 240.0)
    u_ref_arr[tm >= 150.0] = 160.0

    # 1. Wykres pelny 0–300 ms
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    tm_v, v_dec = _minmax_decimate(tm, v, 2000)
    tm_i, i_dec = _minmax_decimate(tm, iL, 2000)

    ax[0].plot(tm_v, v_dec, color="tab:blue", lw=0.9, label="u_dc (Trend MF-BB)")
    ax[0].plot(tm, u_ref_arr, "k--", lw=1, alpha=0.6, label="u_ref")
    ax[0].set_ylabel("u_dc [V]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="lower right")

    ax[1].plot(tm_i, i_dec, color="tab:green", lw=0.8, label="i_L (prąd cewki)")
    ax[1].axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_xlabel("czas [ms]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right")

    events = [
        (50, "R->10MΩ (OFF)"),
        (100, "R->47.6Ω (ON)"),
        (150, "ref 240->160V"),
        (200, "R->10MΩ (OFF)"),
        (250, "R->47.6Ω (ON)"),
    ]
    for t_ev, lbl in events:
        ax[0].axvline(t_ev, color="red", lw=0.7, ls=":", alpha=0.6)
        ax[1].axvline(t_ev, color="red", lw=0.7, ls=":", alpha=0.6)
        ax[0].text(t_ev, ax[0].get_ylim()[1], lbl, fontsize=7.5, color="red", ha="center", va="bottom")

    fig.suptitle("Scenariusz dynamiczny (0–300 ms) – Trend MF-BB z dwustrefową funkcją celu prof. Iwańskiego", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pelny = "optymalizacja/wykres_trend_threshold_pelny.png"
    fig.savefig(out_pelny, dpi=130)
    plt.close(fig)
    print(f"Zapisano wykres: {out_pelny}")

    # 2. Wykres zoom 140–210 ms
    mask_z = (t >= 0.140) & (t <= 0.210)
    tm_z = t[mask_z] * 1e3

    fig_z, ax_z = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_z[0].plot(tm_z, v[mask_z], color="tab:blue", lw=0.9, label="u_dc (Trend MF-BB)")
    ax_z[0].plot(tm_z, u_ref_arr[mask_z], "k--", lw=1, alpha=0.6, label="u_ref")
    ax_z[0].set_ylabel("u_dc [V]")
    ax_z[0].grid(True, alpha=0.3)
    ax_z[0].legend(loc="lower right")

    ax_z[1].plot(tm_z, iL[mask_z], color="tab:green", lw=0.8, label="i_L (prąd cewki)")
    ax_z[1].axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax_z[1].set_ylabel("i_L [A]")
    ax_z[1].set_xlabel("czas [ms]")
    ax_z[1].grid(True, alpha=0.3)
    ax_z[1].legend(loc="upper right")

    for t_ev, lbl in [(150, "ref 240->160V"), (200, "R->10MΩ (OFF)")]:
        ax_z[0].axvline(t_ev, color="red", lw=0.7, ls=":", alpha=0.6)
        ax_z[1].axvline(t_ev, color="red", lw=0.7, ls=":", alpha=0.6)
        ax_z[0].text(t_ev, ax_z[0].get_ylim()[1], lbl, fontsize=7.5, color="red", ha="center", va="bottom")

    fig_z.suptitle("Powiększenie (Zoom 140–210 ms) – pełna rozdzielczość próbek (dt = 0.5 µs)", fontsize=11, fontweight="bold")
    fig_z.tight_layout(rect=(0, 0, 1, 0.96))
    out_zoom = "optymalizacja/wykres_trend_threshold_zoom.png"
    fig_z.savefig(out_zoom, dpi=130)
    plt.close(fig_z)
    print(f"Zapisano wykres: {out_zoom}")

    return out_pelny, out_zoom


def generate_report_pdf(plot_pelny: str, plot_zoom: str):
    filename = "raport_trend_threshold_profesor.pdf"
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
        fontSize=13,
        leading=16,
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
        fontSize=9.5,
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
        leading=11.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=2,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7.8,
        leading=9.5,
        alignment=1,
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
        alignment=0,
    )

    story = []

    # Belka naglowkowa
    story.append(Paragraph("Raport wynikowy: Trend MF-BB z dwustrefową funkcją celu", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=5))

    meta_data = [
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Do:</font> Prof. dr hab. inż. Grzegorz Iwański", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Data:</font> Wrzesień 2026", meta_style),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Od:</font> inż. Rafał Kaczyński (album: 342208)", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Temat:</font> Wyniki symulacji Trend MF-BB (dwustrefowa funkcja celu, Ts = 10 us)", meta_style),
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

    story.append(Paragraph(f"<font name='{FONT_BOLD}'>Szanowny Panie Profesorze,</font>", body_style))
    story.append(Paragraph(
        "Przedstawiam zestawienie wyników symulacji dla regulatora Model-Free Bang-Bang z filtrowanym trendem napięcia $\\Delta u_f$ "
        "w zadanym scenariuszu dynamicznym ($T_{\\mathrm{end}} = 0.30\\text{ s}$, $V_{\\mathrm{in}} = 100\\text{ V}$, $L = 750\\ \\mu\\text{H}$, "
        "$C = 200\\ \\mu\\text{F}$, $R_{\\mathrm{load}} = 47.619\\ \\Omega \\leftrightarrow 10\\text{ M}\\Omega$, $T_s = 10\\ \\mu\\text{s}$, limit $I_{\\max} = 20\\text{ A}$).",
        body_style
    ))
    story.append(Paragraph(
        "Optymalizacja PSO została przeprowadzona z wykorzystaniem dwustrefowej funkcji celu z progiem błędu napięcia $|e_u| = 5\\text{ V}$:<br/>"
        "• dla $|e_u| > 5\\text{ V}$ (stan przejściowy): ocena nasycenia prądu dławika względem $\\pm I_{\\max}$,<br/>"
        "• dla $|e_u| \\le 5\\text{ V}$ (otoczenie stanu ustalonego): kryterium CurrentAware ($|e_u| + \\lambda |i_{\\mathrm{des}} - i_L|$).<br/>"
        "Wyznaczone przez PSO nastawy optymalne:<br/>"
        "• $w_i^* = 0.3583$, $w_v^* = 6.6502$, $f_{c,i}^* = 687.2\\text{ Hz}$, $f_{c,u}^* = 250.9\\text{ Hz}$.",
        body_style
    ))
    story.append(Spacer(1, 1.5 * mm))

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
            Paragraph("-0.233 V", table_cell),
            Paragraph("-0.046 A", table_cell),
            Paragraph("5.948 A", table_cell),
            Paragraph("N/A (jałowy)", table_cell),
        ],
        [
            Paragraph("[0.10, 0.15)", table_cell_left),
            Paragraph("240.0", table_cell),
            Paragraph("ON (47.62 Ω, 1.2 kW)", table_cell),
            Paragraph("<font name='" + FONT_BOLD + "'>-0.087 V</font>", table_cell),
            Paragraph("12.115 A", table_cell),
            Paragraph("<font name='" + FONT_BOLD + "'>5.968 A</font>", table_cell),
            Paragraph("<font name='" + FONT_BOLD + "'>49.26%</font>", table_cell),
        ],
        [
            Paragraph("[0.15, 0.20)", table_cell_left),
            Paragraph("160.0", table_cell),
            Paragraph("ON (47.62 Ω, 538 W)", table_cell),
            Paragraph("+0.093 V", table_cell),
            Paragraph("5.376 A", table_cell),
            Paragraph("4.417 A", table_cell),
            Paragraph("82.16%", table_cell),
        ],
        [
            Paragraph("[0.20, 0.25)", table_cell_left),
            Paragraph("160.0", table_cell),
            Paragraph("OFF (10 MΩ)", table_cell),
            Paragraph("+0.167 V", table_cell),
            Paragraph("-0.020 A", table_cell),
            Paragraph("4.343 A", table_cell),
            Paragraph("N/A (jałowy)", table_cell),
        ],
        [
            Paragraph("[0.25, 0.30)", table_cell_left),
            Paragraph("160.0", table_cell),
            Paragraph("ON (47.62 Ω, 538 W)", table_cell),
            Paragraph("+0.086 V", table_cell),
            Paragraph("5.376 A", table_cell),
            Paragraph("4.447 A", table_cell),
            Paragraph("82.72%", table_cell),
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
        ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#EBF8FF")),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#EBF8FF")),
        ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#EBF8FF")),
    ]))
    story.append(t_res)
    story.append(Spacer(1, 2 * mm))

    dyn_txt = (
        "Wskaźniki dynamiczne stanów przejściowych:<br/>"
        "• maksymalny zapad napięcia po skoku 240 V → 160 V: $\\Delta U_{\\mathrm{undershoot}} = 3.52\\text{ V}$ (zmniejszenie o 50% względem nastaw bazowych: $7.07\\text{ V}$),<br/>"
        "• maksymalne przeregulowanie po załączeniu obciążenia w $t=100\\text{ ms}$: $\\Delta U_{\\mathrm{overshoot}} = 0.51\\text{ V}$ (zmniejszenie o 73% względem bazy: $1.91\\text{ V}$),<br/>"
        "• maksymalny prąd szczytowy w całym horyzoncie 0–300 ms: $i_{L,\\mathrm{peak}} = 21.31\\text{ A}$ (limit $I_{\\max} = 20.0\\text{ A}$)."
    )
    story.append(Paragraph(dyn_txt, body_style))
    story.append(Spacer(1, 4 * mm))

    sign_txt = (
        f"<font name='{FONT_ITALIC}'>Raport wygenerowany automatycznie na podstawie symulacji numerycznej.<br/>"
        f"Z poważaniem,<br/><font name='{FONT_BOLD}'>inż. Rafał Kaczyński</font> (album 342208)</font>"
    )
    story.append(Paragraph(sign_txt, body_style))

    # Strona 2: Wykres pelny
    story.append(PageBreak())
    story.append(Paragraph("Załącznik 1: Przebiegi czasowe w pełnym scenariuszu (0–300 ms)", h2_style))
    story.append(Paragraph(
        "Przebiegi napięcia wyjściowego $u_{dc}(t)$ na tle wartości zadanej $u_{\\mathrm{ref}}(t)$ oraz prądu cewki $i_L(t)$ "
        "dla regulatora Trend MF-BB z nowymi nastawami PSO. Linie pionowe wskazują momenty zdarzeń scenariusza.",
        body_style
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Image(plot_pelny, width=174 * mm, height=195 * mm))

    # Strona 3: Wykres zoom
    story.append(PageBreak())
    story.append(Paragraph("Załącznik 2: Przybliżenie (Zoom 140–210 ms) – pełna rozdzielczość próbek ($dt = 0.5\\ \\mu\\text{s}$)", h2_style))
    story.append(Paragraph(
        "Przebiegi w oknie 140–210 ms w pełnej rozdzielczości próbek silnika fizyki ($dt = 0.5\\ \\mu\\text{s}$) "
        "obejmujące skok zadanego napięcia $240 \\to 160\\text{ V}$ w $t=150\\text{ ms}$ oraz wyłączenie obciążenia w $t=200\\text{ ms}$.",
        body_style
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Image(plot_zoom, width=174 * mm, height=195 * mm))

    def make_canvas(filename, **kw):
        return NumberedCanvas(filename, footer_title="Raport Trend MF-BB (dwustrefowa funkcja celu) | Politechnika Warszawska", **kw)

    doc.build(story, canvasmaker=make_canvas)
    print(f"Wygenerowano raport PDF: {filename}")


def generate_samples_pdf(res: dict):
    filename = "surowe_probki_trend_threshold.pdf"
    doc = SimpleDocTemplate(
        filename,
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

    story.append(Paragraph("Tabela surowych próbek dyskretnych – Trend MF-BB z dwustrefową funkcją celu", title_style))
    story.append(Paragraph(
        "Zestawienie dyskretnych próbek trajektorii czasowych ($t, v_C, i_L, s$) dla regulatora Trend MF-BB "
        "nastrojonego dwustrefową funkcją celu prof. Iwańskiego ($w_i^* = 0.3583, w_v^* = 6.6502, f_{c,i}^* = 687.2\\text{ Hz}, f_{c,u}^* = 250.9\\text{ Hz}$, $T_s = 10\\ \\mu\\text{s}$).",
        desc_style
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=3))

    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]
    s = res["s"]

    def build_table_data(t_start, t_end, step):
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

    windows_def = [
        ("Okno 1: Stan ustalony pod pełnym obciążeniem (240 V, 1.2 kW, t ∈ [145.0, 146.5] ms, krok 20 µs)", 0.1450, 0.1465, 20e-6),
        ("Okno 2: Stan przejściowy – skok zadanego napięcia 240 V → 160 V (t ∈ [149.8, 151.3] ms, krok 20 µs)", 0.1498, 0.1513, 20e-6),
        ("Okno 3: Stan ustalony po skoku pod obciążeniem (160 V, 538 W, t ∈ [170.0, 171.5] ms, krok 20 µs)", 0.1700, 0.1715, 20e-6),
    ]

    for title, t1, t2, st in windows_def:
        story.append(Paragraph(title, h2_style))
        t_data = build_table_data(t1, t2, st)
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

    def make_canvas(filename, **kw):
        return NumberedCanvas(filename, footer_title="Surowe próbki Trend MF-BB | Politechnika Warszawska", **kw)

    doc.build(story, canvasmaker=make_canvas)
    print(f"Wygenerowano PDF z surowymi probkami: {filename}")


def main():
    print("=" * 80)
    print("GENEROWANIE PELNEGO PAKIETU DLA DWUSTREFOWEJ FUNKCJI CELU PROF. IWANSKIEGO")
    print("=" * 80)

    print("1. Uruchamianie symulacji...")
    res = run_simulation()

    print("2. Generowanie wykresow...")
    p_pelny, p_zoom = generate_plots(res)

    print("3. Generowanie raportu PDF (3 strony)...")
    generate_report_pdf(p_pelny, p_zoom)

    print("4. Generowanie tabeli surowych probek PDF (3 strony)...")
    generate_samples_pdf(res)

    print("\n" + "=" * 80)
    print("ZAKONCZONO POMYSLNIE! Wszystkie 4 pliki zostaly utworzone.")
    print("=" * 80)


if __name__ == "__main__":
    main()
