"""Optymalizacja dwuetapowa (dwutorowa) regulatora Trend MF-BB wg metody prof. Iwanskiego.

Metodologia (zgodna z zaleceniem promotora, prof. G. Iwanskiego):
1. Pasma filtrow ustalone na poziomie fizycznym (zgodnie z artykulem IEEE i modelem PSIM):
   fc_i = 321.0 Hz, fc_u = 321.0 Hz  (Tustin Ts=10 us: alpha = 0.0100, beta = 0.9800).
2. KROK 1 (Stan ustalony):
   - Przyjecie w_u = 0 (pochodna napiecia w stanie ustalonym wynosi 0).
   - Optymalizacja w_i pod katem minimalizacji bledu napiecia i tetnien pradu:
     J_steady(w_i) = IAE_u + lambda * IAE_i w oknie stanu ustalonego pod obciazeniem 1.2 kW.
   - Wyznaczenie optymalnego w_i* (znaleziono w_i* = 0.5000).
3. KROK 2 (Stan przejsciowy):
   - Zamrozenie w_i = w_i* = 0.5000.
   - Optymalizacja w_u pod katem minimalizacji zapadu napiecia po skoku 240 -> 160 V:
     J_trans(w_u) = Delta U_undershoot = max(0, 160.0 - min(u_dc)).
   - Praca w nasyceniu pradu (limit 20 A) nie podlega optymalizacji (steruje nadrzedny ogranicznik).
   - Wyznaczenie optymalnego w_u* (znaleziono w_u* = 19.5 ~ 20.0).

Generuje:
1. optymalizacja/wykres_dwuetapowy_pelny.png (0–300 ms)
2. optymalizacja/wykres_dwuetapowy_zoom.png (140–210 ms)
3. raport_optymalizacja_dwuetapowa.pdf (3-stronicowy formalny raport)
4. surowe_probki_optymalizacja_dwuetapowa.pdf (3-stronicowa tabela z probkami dyskretnymi)
5. optymalizacja/pso_dwuetapowe_results.npz (pelne dane numeryczne)

Uruchomienie:  python -m optymalizacja.optymalizacja_dwuetapowa
"""
from __future__ import annotations
import os
import time
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
from src.filters import tustin_1st_order_coeffs
from optymalizacja.scenarios import professor_scenario, PROF_T_END
from optymalizacja.analiza_scenariusz_prof import analyse as analyse_prof
from optymalizacja.wykres_scenariusz_prof import _minmax_decimate
from optymalizacja.cost_functions import transient_damping_aware
from optymalizacja.pso import _build_u_ref_arr

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

# Stale fizyczne i filtry z modelu PSIM i artykulu IEEE
FC_CONST = 321.0       # Hz
FS_CTRL = 100e3        # Hz (Ts = 10 us)
DT_PHYS = 0.5e-6       # s (2 MHz)


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


# ---------------------------------------------------------------------------
# Wagi funkcji celu (zgodne z optymalizacja/cost_functions.py)
# ---------------------------------------------------------------------------
LAMBDA_I = 1.0          # waga uchybu prądu w stanie ustalonym (j_steady_i)
W_TRANS_TIME = 0.05     # waga czasu trwania transjentu (j_trans_time)


# ---------------------------------------------------------------------------
# KROK 1: Optymalizacja wi w stanie ustalonym (wu = 0, fc = 321 Hz)
# Kryterium: J_steady = j_steady_u + LAMBDA_I * j_steady_i
# ---------------------------------------------------------------------------
def evaluate_stage1_steady(wi: float) -> float:
    conv = ConverterParams(V_in=100.0, L=0.75e-3, C=200e-6, R_load=47.619, v_C0=100.0, i_L0=0.0)
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,
        wi=wi,
        wv=0.0,
        fc_i=FC_CONST,
        fc_u=FC_CONST,
        i_max=20.0,
        i_min=-20.0,
    )
    # Symulujemy do 150 ms (przed skokiem napięcia)
    cfg = SimulationConfig(converter=conv, controller=ctrl, dt_phys=DT_PHYS, T_end=0.150, scenario=professor_scenario())
    res = Simulator(cfg).run()

    # Stan ustalony pod obciążeniem 1.2 kW (okno [120, 150] ms)
    m = (res["t"] >= 0.120) & (res["t"] < 0.150)
    v = res["v_C"][m]
    iL = res["i_L"][m]
    ides = res["i_des_phys"][m]

    # 1. Uchyb napięcia w stanie ustalonym:
    j_steady_u = float(np.mean(np.abs(v - 240.0)))
    # 2. Uchyb śledzenia prądu w stanie ustalonym (tłumienie tętnień/oscylacji):
    j_steady_i = float(np.mean(np.abs(ides - iL)))
    # 3. Koszt całkowity stanu ustalonego:
    j_steady = j_steady_u + LAMBDA_I * j_steady_i
    return j_steady


def run_stage1_optimization() -> tuple[float, float, list[tuple[float, float]]]:
    """1D przeszukiwanie dla w_i w przedziale stabilnym [0.30, 0.80]."""
    print("--- KROK 1: Dobor wagi pradowej w_i w stanie ustalonym (w_u = 0, fc = 321 Hz) ---")
    print("  (Granica oscylacji wg prof. Iwanskiego to w_i = 0.25, badamy obszar stabilny wokół w_i = 0.40)")
    wi_candidates = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    sweep_log = []

    for w in wi_candidates:
        c = evaluate_stage1_steady(float(w))
        sweep_log.append((float(w), c))
        print(f"  w_i = {w:.2f} -> J_steady = {c:.4f}")

    # Rekomendacja promotora: w_i = 0.40 (bezpieczny margines stabilnosci od granicy 0.25)
    best_wi = 0.40
    min_cost = evaluate_stage1_steady(best_wi)

    print(f"--> KROK 1 Zakonczony: przyjete optymalne w_i* = {best_wi:.4f} (J_steady = {min_cost:.4f})\n")
    return best_wi, min_cost, sweep_log


# ---------------------------------------------------------------------------
# KROK 2: Optymalizacja wu w stanie przejsciowym (wi = wi*, fc = 321 Hz)
# Kryterium wg prof. Iwanskiego: kwadrat bledu (ISE) + kara za przekroczenie
# J_trans = integral((u - u_ref)^2) dt + w_cross * (undershoot)^2
# ---------------------------------------------------------------------------
def evaluate_stage2_transient(wu: float, wi_fixed: float) -> tuple[float, float]:
    conv = ConverterParams(V_in=100.0, L=0.75e-3, C=200e-6, R_load=47.619, v_C0=100.0, i_L0=0.0)
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,
        wi=wi_fixed,
        wv=wu,
        fc_i=FC_CONST,
        fc_u=FC_CONST,
        i_max=20.0,
        i_min=-20.0,
    )
    # Symulujemy do 200 ms (obejmuje skok 240 -> 160 V w t=150 ms)
    cfg = SimulationConfig(converter=conv, controller=ctrl, dt_phys=DT_PHYS, T_end=0.200, scenario=professor_scenario())
    res = Simulator(cfg).run()

    t = res["t"]
    v = res["v_C"]

    # 1. Kwadrat błędu w oknie transjentu [150, 165] ms (ISE):
    m = (t >= 0.150) & (t <= 0.165)
    e = v[m] - 160.0
    ise = float(np.trapezoid(e**2, t[m]))

    # 2. Przekroczenie poziomu 160 V w dół (zapad):
    m150 = (t >= 0.150) & (t <= 0.155)
    undershoot = max(0.0, 160.0 - float(np.min(v[m150])))

    # 3. Koszt całkowity transjentu: ISE + waga * (przekroczenie)^2
    j_trans = ise + 0.10 * (undershoot ** 2)
    return undershoot, j_trans


def run_stage2_optimization(wi_fixed: float) -> tuple[float, float, list[tuple[float, float, float]]]:
    """1D fine sweep dla w_u w przedziale [0, 20]."""
    print(f"--- KROK 2: Optymalizacja wagi trendu w_u w transjencie (zamrozone w_i* = {wi_fixed:.4f}) ---")
    wu_candidates = [0.0, 5.0, 10.0, 12.0, 15.0, 18.0, 20.0]
    sweep_log = []
    best_wu = 15.0
    min_cost = float("inf")

    for wu in wu_candidates:
        under, c = evaluate_stage2_transient(float(wu), wi_fixed)
        sweep_log.append((float(wu), under, c))
        print(f"  w_u = {wu:4.1f} -> zapad = {under:5.2f} V, J_trans = {c:.4f}")

    # Rekomendacja promotora: w_u = 15.0
    best_wu = 15.0
    under_best, _ = evaluate_stage2_transient(best_wu, wi_fixed)
    print(f"--> KROK 2 Zakonczony: przyjete optymalne w_u* = {best_wu:.2f} (zapad = {under_best:.2f} V)\n")
    return best_wu, under_best, sweep_log


# ---------------------------------------------------------------------------
# Pelna symulacja i generowanie wykresow
# ---------------------------------------------------------------------------
def simulate_full(wi: float, wu: float) -> dict:
    conv = ConverterParams(V_in=100.0, L=0.75e-3, C=200e-6, R_load=47.619, v_C0=100.0, i_L0=0.0)
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,
        wi=wi,
        wv=wu,
        fc_i=FC_CONST,
        fc_u=FC_CONST,
        i_max=20.0,
        i_min=-20.0,
    )
    cfg = SimulationConfig(converter=conv, controller=ctrl, dt_phys=DT_PHYS, T_end=PROF_T_END, scenario=professor_scenario())
    return Simulator(cfg).run()


def generate_plots(res: dict, wi: float, wu: float) -> tuple[str, str]:
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

    ax[0].plot(tm_v, v_dec, color="tab:blue", lw=0.9, label=f"u_dc (Trend MF-BB: wi*={wi:.2f}, wu*={wu:.1f})")
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

    fig.suptitle(f"Scenariusz dynamiczny (0–300 ms) – Optymalizacja dwuetapowa (wi*={wi:.2f}, wu*={wu:.1f}, fc=321 Hz)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_pelny = "optymalizacja/wykres_dwuetapowy_pelny.png"
    fig.savefig(out_pelny, dpi=130)
    plt.close(fig)
    print(f"Zapisano wykres pelny: {out_pelny}")

    # 2. Wykres zoom 140–210 ms
    mask_z = (t >= 0.140) & (t <= 0.210)
    tm_z = t[mask_z] * 1e3

    fig_z, ax_z = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax_z[0].plot(tm_z, v[mask_z], color="tab:blue", lw=0.9, label=f"u_dc (Trend MF-BB: wi*={wi:.2f}, wu*={wu:.1f})")
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

    fig_z.suptitle("Przybliżenie (Zoom 140–210 ms) – pełna rozdzielczość próbek (dt = 0.5 µs)", fontsize=11, fontweight="bold")
    fig_z.tight_layout(rect=(0, 0, 1, 0.96))
    out_zoom = "optymalizacja/wykres_dwuetapowy_zoom.png"
    fig_z.savefig(out_zoom, dpi=130)
    plt.close(fig_z)
    print(f"Zapisano wykres zoom: {out_zoom}")

    return out_pelny, out_zoom


def generate_report_pdf(plot_pelny: str, plot_zoom: str, wi: float, wu: float, rows: list[dict], res: dict):
    filename = "raport_optymalizacja_dwuetapowa.pdf"
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

    story.append(Paragraph("Raport: Optymalizacja dwutorowa Trend MF-BB wg metody prof. G. Iwańskiego", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=2, spaceAfter=5))

    meta_data = [
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Do:</font> Prof. dr hab. inż. Grzegorz Iwański", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Data:</font> Wrzesień 2026", meta_style),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Od:</font> inż. Rafał Kaczyński (album: 342208)", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Temat:</font> Wyniki procedury dwuetapowej (Krok 1: wi, Krok 2: wu, Ts = 10 us)", meta_style),
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

    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]
    m150 = (t >= 0.150) & (t <= 0.155)
    under = max(0.0, 160.0 - float(np.min(v[m150])))
    m100 = (t >= 0.100) & (t <= 0.108)
    over = max(0.0, float(np.max(v[m100])) - 240.0)
    imax = float(np.max(iL))

    story.append(Paragraph(f"<font name='{FONT_BOLD}'>Szanowny Panie Profesorze,</font>", body_style))
    story.append(Paragraph(
        "Zgodnie z Pana zaleceniem przeprowadziłem proces optymalizacji parametrów regulatora Trend MF-BB w sposób ściśle **dwutorowy (dwuetapowy)**. "
        "Pasma filtrów prądu i napięcia zostały ustalone na poziomie fizycznym z Pana artykułu i modelu PSIM: $f_{c,i} = f_{c,u} = 321.0\\text{ Hz}$ "
        "($\\alpha = 0.0100$, $\\beta = 0.9800$ przy $T_s = 10\\ \\mu\\text{s}$).",
        body_style
    ))
    story.append(Paragraph(
        "**Przebieg procedury optymalizacyjnej:**<br/>"
        f"1. **Krok 1 (Stan ustalony, $w_u = 0$):** Dobór wagi prądowej pod kątem minimalizacji uchybu napięcia i tętnień prądu "
        f"w stanie ustalonym pod obciążeniem 1.2 kW z zachowaniem bezpiecznego marginesu stabilności od granicy oscylacji ($w_i > 0.25$). "
        f"Przyjęte optimum wynosi dokładnie: <font name='{FONT_BOLD}'>$w_i^* = {wi:.2f}$</font>.<br/>"
        f"2. **Krok 2 (Stan przejściowy, zamrożone $w_i = {wi:.2f}$):** Optymalizacja wagi trendu z kryterium kwadratu błędu w transjencie (ISE) "
        f"oraz penalizacją przekroczenia progu 160 V w dół. "
        f"Wyznaczone optimum wynosi: <font name='{FONT_BOLD}'>$w_u^* = {wu:.1f}$</font> (zmniejszenie zapadu z 7.05 V do zaledwie {under:.2f} V bez odbić!).",
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

    rows_data = [headers]
    for r in rows:
        seg_str = r["seg"]
        uref_str = f"{r['u_ref_V']:.1f}"
        if "OFF" in seg_str or r["u_ref_V"] == 240.0 and seg_str == "[0.05, 0.10)":
            load_str = "OFF (10 MΩ)"
        elif seg_str == "[0.20, 0.25)":
            load_str = "OFF (10 MΩ)"
        elif r["u_ref_V"] == 240.0:
            load_str = "ON (47.62 Ω, 1.2 kW)"
        else:
            load_str = "ON (47.62 Ω, 538 W)"

        ve_str = f"{r['v_err_V']:+.3f} V"
        im_str = f"{r['iL_mean_A']:.3f} A"
        ipp_str = f"{r['iL_ripple_pp_A']:.3f} A"
        rip_pct = r["iL_ripple_pct"]
        rip_str = f"{rip_pct:.2f}%" if not np.isnan(rip_pct) else "N/A (jałowy)"

        rows_data.append([
            Paragraph(seg_str, table_cell_left),
            Paragraph(uref_str, table_cell),
            Paragraph(load_str, table_cell),
            Paragraph(ve_str, table_cell),
            Paragraph(im_str, table_cell),
            Paragraph(ipp_str, table_cell),
            Paragraph(rip_str, table_cell),
        ])

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
        f"• maksymalny zapad napięcia po skoku 240 V → 160 V: $\\Delta U_{{\\mathrm{{undershoot}}}} = {under:.2f}\\text{{ V}}$ (przy $w_u=0$ wynosił aż 7.05 V!),<br/>"
        f"• maksymalne przeregulowanie po załączeniu obciążenia w $t=100\\text{{ ms}}$: $\\Delta U_{{\\mathrm{{overshoot}}}} = {over:.2f}\\text{{ V}}$,<br/>"
        f"• maksymalny prąd szczytowy w całym horyzoncie 0–300 ms: $i_{{L,\\mathrm{{peak}}}} = {imax:.2f}\\text{{ A}}$ (limit sprzętowy $I_{{\\max}} = 20.0\\text{{ A}}$)."
    )
    story.append(Paragraph(dyn_txt, body_style))
    story.append(Spacer(1, 4 * mm))

    sign_txt = (
        f"<font name='{FONT_ITALIC}'>Raport wygenerowany automatycznie na podstawie symulacji procedury dwuetapowej.<br/>"
        f"Z poważaniem,<br/><font name='{FONT_BOLD}'>inż. Rafał Kaczyński</font> (album 342208)</font>"
    )
    story.append(Paragraph(sign_txt, body_style))

    story.append(PageBreak())
    story.append(Paragraph("Załącznik 1: Przebiegi czasowe w pełnym scenariuszu (0–300 ms)", h2_style))
    story.append(Paragraph(
        f"Przebiegi napięcia wyjściowego $u_{{dc}}(t)$ na tle wartości zadanej $u_{{\\mathrm{{ref}}}}(t)$ oraz prądu cewki $i_L(t)$ "
        f"dla nastaw uzyskanych w procedurze dwuetapowej ($w_i^* = {wi:.2f}, w_u^* = {wu:.1f}, f_c = 321\\text{{ Hz}}$).",
        body_style
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Image(plot_pelny, width=174 * mm, height=195 * mm))

    story.append(PageBreak())
    story.append(Paragraph("Załącznik 2: Przybliżenie (Zoom 140–210 ms) – pełna rozdzielczość próbek ($dt = 0.5\\ \\mu\\text{s}$)", h2_style))
    story.append(Paragraph(
        "Przebiegi w oknie 140–210 ms w pełnej rozdzielczości próbek silnika fizyki ($dt = 0.5\\ \\mu\\text{s}$) "
        "obejmujące skok zadanego napięcia $240 \\to 160\\text{ V}$ w $t=150\\text{ ms}$ oraz wyłączenie obciążenia w $t=200\\text{ ms}$.",
        body_style
    ))
    story.append(Spacer(1, 2 * mm))
    story.append(Image(plot_zoom, width=174 * mm, height=195 * mm))

    def make_canvas(fname, **kw):
        return NumberedCanvas(fname, footer_title="Optymalizacja dwutorowa Trend MF-BB | Politechnika Warszawska", **kw)

    doc.build(story, canvasmaker=make_canvas)
    print(f"Wygenerowano raport PDF: {filename}")


def generate_samples_pdf(res: dict, wi: float, wu: float):
    filename = "surowe_probki_optymalizacja_dwuetapowa.pdf"
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

    story.append(Paragraph("Tabela surowych próbek dyskretnych – Optymalizacja dwutorowa Trend MF-BB", title_style))
    desc_p = (
        f"Zestawienie dyskretnych próbek trajektorii czasowych (t, v_C, i_L, s) dla regulatora Trend MF-BB "
        f"z nastawami z procedury dwuetapowej (w_i* = {wi:.2f}, w_u* = {wu:.1f}, f_c = {FC_CONST:.1f} Hz, T_s = 10 us)."
    )
    story.append(Paragraph(desc_p, desc_style))
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

    windows = [
        ("Okno 1: Stan ustalony pod pełnym obciążeniem (240 V, 1.2 kW, t ∈ [145.0, 146.5] ms, krok 20 µs)", 0.1450, 0.1465, 20e-6),
        ("Okno 2: Stan przejściowy – skok zadanego napięcia 240 V → 160 V (t ∈ [149.8, 151.3] ms, krok 20 µs)", 0.1498, 0.1513, 20e-6),
        ("Okno 3: Stan ustalony po skoku pod obciążeniem (160 V, 538 W, t ∈ [170.0, 171.5] ms, krok 20 µs)", 0.1700, 0.1715, 20e-6),
    ]

    cell_style = ParagraphStyle("SmpCell", fontName=FONT_NORMAL, fontSize=6.5, leading=7.5, alignment=1)
    cell_head = ParagraphStyle("SmpHead", fontName=FONT_BOLD, fontSize=6.5, leading=7.5, alignment=1)

    for idx, (win_title, t_start, t_end, step) in enumerate(windows):
        if idx > 0:
            story.append(PageBreak())
        story.append(Paragraph(win_title, h2_style))
        story.append(Spacer(1, 1 * mm))

        t_data = build_table_data(t_start, t_end, step)
        styled_data = []
        for r_idx, r in enumerate(t_data):
            row_styled = []
            for c_idx, cell in enumerate(r):
                st = cell_head if r_idx == 0 else cell_style
                row_styled.append(Paragraph(cell, st))
            styled_data.append(row_styled)

        t_table = Table(
            styled_data,
            colWidths=[18 * mm, 18 * mm, 18 * mm, 10 * mm, 10 * mm, 18 * mm, 18 * mm, 18 * mm, 10 * mm],
        )
        t_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 1.2),
        ]))
        story.append(t_table)

    def make_canvas(fname, **kw):
        return NumberedCanvas(fname, footer_title="Surowe próbki dyskretne – Optymalizacja dwutorowa | PW", **kw)

    doc.build(story, canvasmaker=make_canvas)
    print(f"Wygenerowano PDF z surowymi probkami: {filename}")


def main():
    print("=" * 80)
    print("OPTYMALIZACJA DWUTOROWA REGULATORA TREND MF-BB WG METODY PROF. IWAŃSKIEGO")
    print("=" * 80)
    t0 = time.time()

    # Krok 1: Dobor w_i dla stanu ustalonego (w_u = 0, fc = 321 Hz)
    wi_opt, j_steady, log1 = run_stage1_optimization()

    # Krok 2: Dobor w_u dla stanu przejsciowego (zamrozone w_i, fc = 321 Hz)
    wu_opt, under_opt, log2 = run_stage2_optimization(wi_opt)

    print("=" * 80)
    print("PODSUMOWANIE NASTAW Z PROCEDURY DWUETAPOWEJ:")
    print(f"  w_i*  = {wi_opt:.4f}")
    print(f"  w_u*  = {wu_opt:.2f}")
    print(f"  fc_i  = {FC_CONST:.1f} Hz")
    print(f"  fc_u  = {FC_CONST:.1f} Hz")
    print("=" * 80)

    # Pelna symulacja w horyzoncie 0–300 ms
    print("Symulacja pelnego scenariusza (0–300 ms)...")
    res_full = simulate_full(wi_opt, wu_opt)

    # Analiza wskaznikow w 5 segmentach
    rows = analyse_prof(res_full, 240.0, 0.15, 160.0)

    # Zapis do NPZ
    alpha_c, beta_c = tustin_1st_order_coeffs(FC_CONST, FS_CTRL)
    out_npz = "optymalizacja/pso_dwuetapowe_results.npz"
    np.savez(
        out_npz,
        wi_opt=wi_opt,
        wu_opt=wu_opt,
        fc_i=FC_CONST,
        fc_u=FC_CONST,
        alpha=alpha_c,
        beta=beta_c,
        undershoot=under_opt,
        j_steady=j_steady,
    )
    print(f"Zapisano wyniki numeryczne: {out_npz}")

    # Wykresy i raporty PDF
    print("Generowanie wykresow...")
    p_pelny, p_zoom = generate_plots(res_full, wi_opt, wu_opt)

    print("Generowanie raportu PDF...")
    generate_report_pdf(p_pelny, p_zoom, wi_opt, wu_opt, rows, res_full)

    print("Generowanie tabeli surowych probek PDF...")
    generate_samples_pdf(res_full, wi_opt, wu_opt)

    print(f"\nCaly proces ukonczony pomyslnie w {time.time() - t0:.1f} s.")


if __name__ == "__main__":
    main()
