"""Badanie wrażliwosci dwutorowego algorytmu optymalizacyjnego Trend MF-BB.

Zadanie zlecone przez promotora (prof. G. Iwanskiego):
1. WRAZLIWOSC KROKU 1 (stan ustalony: wi* oraz fc_i*):
   Jak zmiana relacji miedzy bledem napiecia a bledem pradu (waga lambda_i w J1)
   wplywa na optymalne nastawy wi* i fc_i*, uchyb napiecia oraz tetnienia pradu.
   Wzor KROK 1:
     J_1 = IAE_u + lambda_i * IAE_i
     gdzie IAE_u = mean(|u_dc - 240V|), IAE_i = mean(|i_des - i_L|) w stanie ustalonym 1.2 kW.

2. WZOR KROKU 2 (stan przejsciowy: wu* oraz fc_u*):
   Formalny wzor funkcji celu transjentu (skok 240 -> 160 V w t=150 ms):
     J_2 = ISE_trans + w_cross * (Delta U_undershoot)^2
     gdzie:
       ISE_trans = integral_{t=150ms}^{165ms} (u_dc(t) - 160V)^2 dt  (kwadrat bledu)
       Delta U_undershoot = max(0, 160V - min(u_dc))                 (kara za przekroczenie w dol)
       w_cross = waga kary za zapad (nominalnie 0.10)

3. WRAZLIWOSC KROKU 2:
   a) Wplyw wagi w_cross (wielkosci kary za przekroczenie) na optymalne wu* i fc_u*.
   b) Wplyw wyniku Kroku 1 (wartosci wi* z Kroku 1) na wynik Kroku 2 (optymalne wu*).

Generuje:
- optymalizacja/wykres_badanie_wrazliwosci.png
- optymalizacja/badanie_wrazliwosci_results.npz
- pelne zestawienie tabelaryczne w konsoli

Uruchomienie:  python -m optymalizacja.badanie_wrazliwosci
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
from optymalizacja.scenarios import professor_scenario, PROF_T_END

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

CONV = ConverterParams(V_in=100.0, L=0.75e-3, C=200e-6, R_load=47.619, v_C0=100.0, i_L0=0.0)
DT_PHYS = 0.5e-6
FC_NOMINAL = 321.0


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
# KROK 1: 2D PSO dla (w_i, fc_i) przy roznych lambda_i
# ---------------------------------------------------------------------------
def opt_step1_2d_pso(lam: float, seed: int = 42) -> tuple[float, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    lb = np.array([np.log10(0.20), np.log10(150.0)])
    ub = np.array([np.log10(0.80), np.log10(1000.0)])
    dim = 2
    N_PART = 10
    N_IT = 14
    v_max = (ub - lb) * 0.25

    X = rng.uniform(lb, ub, size=(N_PART, dim))
    X[0] = np.array([np.log10(0.40), np.log10(321.0)])
    V = rng.uniform(-v_max, v_max, size=(N_PART, dim))

    def eval_pt(pt):
        w = float(10**pt[0])
        f = float(10**pt[1])
        ctrl = TrendMFBBParams(
            u_ref=240.0,
            T_s_ctrl=10e-6,
            wi=w,
            wv=0.0,
            fc_i=f,
            fc_u=FC_NOMINAL,
            i_max=20.0,
            i_min=-20.0,
        )
        res = Simulator(SimulationConfig(converter=CONV, controller=ctrl, dt_phys=DT_PHYS, T_end=0.150, scenario=professor_scenario())).run()
        m = (res["t"] >= 0.120) & (res["t"] < 0.150)
        v = res["v_C"][m]
        iL = res["i_L"][m]
        ides = res["i_des_phys"][m]
        iu = float(np.mean(np.abs(v - 240.0)))
        ii = float(np.mean(np.abs(ides - iL)))
        cost = iu + lam * ii
        return cost, iu, ii, float(np.ptp(iL))

    fits = [eval_pt(x) for x in X]
    F = np.array([fit[0] for fit in fits])
    pbest_X = X.copy()
    pbest_F = F.copy()
    g_idx = int(np.argmin(pbest_F))
    gbest_F = float(pbest_F[g_idx])
    gbest_X = pbest_X[g_idx].copy()

    for it in range(1, N_IT + 1):
        w_iner = 0.9 - 0.5 * (it - 1) / max(1, N_IT - 1)
        r1, r2 = rng.random((N_PART, dim)), rng.random((N_PART, dim))
        V = np.clip(w_iner * V + 1.5 * r1 * (pbest_X - X) + 1.5 * r2 * (gbest_X[None, :] - X), -v_max, v_max)
        X = np.clip(X + V, lb, ub)
        fits = [eval_pt(x) for x in X]
        F = np.array([fit[0] for fit in fits])
        imp = F < pbest_F
        pbest_X[imp] = X[imp]
        pbest_F[imp] = F[imp]
        g_idx = int(np.argmin(pbest_F))
        if pbest_F[g_idx] < gbest_F:
            gbest_F = float(pbest_F[g_idx])
            gbest_X = pbest_X[g_idx].copy()

    best_w = float(10**gbest_X[0])
    best_f = float(10**gbest_X[1])
    c, iu, ii, ipp = eval_pt(gbest_X)
    return best_w, best_f, iu, ii, ipp, c


def sweep_step1_sensitivity() -> list[dict]:
    print("=" * 80)
    print("1. KROK 1: 2D PSO dla (w_i*, fc_i*) w zależności od lambda_i")
    print("=" * 80)
    lambdas = [0.10, 0.25, 0.50, 1.00, 2.00, 5.00]

    results = []
    for lam in lambdas:
        bw, bf, iu, ii, ipp, c = opt_step1_2d_pso(lam)
        results.append({
            "lam": lam,
            "wi_opt": bw,
            "fc_i_opt": bf,
            "iae_u": iu,
            "iae_i": ii,
            "iL_pp": ipp,
            "cost": c,
        })
        print(f"  lambda_i = {lam:4.2f} -> wi* = {bw:.4f}, fc_i* = {bf:5.1f} Hz | IAE_u = {iu:.4f} V, IAE_i = {ii:.3f} A, iL_pp = {ipp:.2f} A, J1 = {c:.4f}")
    return results


# ---------------------------------------------------------------------------
# KROK 2: Ewaluacja i badanie wrazliwosci na wage w_cross oraz wi* z Kroku 1
# ---------------------------------------------------------------------------
def eval_krok2(wu: float, wi_fixed: float, fci_fixed: float, w_cross: float) -> tuple[float, float, float]:
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,
        wi=wi_fixed,
        wv=wu,
        fc_i=fci_fixed,
        fc_u=FC_NOMINAL,
        i_max=20.0,
        i_min=-20.0,
    )
    cfg = SimulationConfig(converter=CONV, controller=ctrl, dt_phys=DT_PHYS, T_end=0.200, scenario=professor_scenario())
    res = Simulator(cfg).run()

    t = res["t"]
    v = res["v_C"]

    # 1. Kwadrat bledu w transjencie (150-165 ms) - ISE
    m = (t >= 0.150) & (t <= 0.165)
    e = v[m] - 160.0
    ise = float(np.trapezoid(e**2, t[m]))

    # 2. Przekroczenie / zapad ponizej 160V w dol
    m150 = (t >= 0.150) & (t <= 0.155)
    undershoot = max(0.0, 160.0 - float(np.min(v[m150])))

    cost = ise + w_cross * (undershoot ** 2)
    return cost, undershoot, ise


def sweep_step2_wcross_sensitivity(wi_nominal: float, fci_nominal: float) -> list[dict]:
    print("\n" + "=" * 80)
    print(f"2. KROK 2: Wrażliwość na wagę kary za zapad w_cross (dla nominalnych wi={wi_nominal:.4f}, fci={fci_nominal:.1f}Hz)")
    print("=" * 80)
    w_cross_values = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0, 5.0]
    wu_grid = np.linspace(8.0, 24.0, 33)

    results = []
    for wc in w_cross_values:
        best_c = float("inf")
        best_wu = 15.0
        best_under, best_ise = 0.0, 0.0
        for wu in wu_grid:
            c, under, ise = eval_krok2(float(wu), wi_nominal, fci_nominal, wc)
            if c < best_c:
                best_c = c
                best_wu = float(wu)
                best_under, best_ise = under, ise
        results.append({
            "w_cross": wc,
            "wu_opt": best_wu,
            "cost": best_c,
            "undershoot": best_under,
            "ise": best_ise,
        })
        print(f"  w_cross = {wc:4.2f} -> optymalne wu* = {best_wu:4.1f} | zapad = {best_under:.2f} V, ISE = {best_ise:.4f}, J2 = {best_c:.4f}")
    return results


def sweep_step2_cross_influence(res_step1: list[dict], w_cross: float = 0.10) -> list[dict]:
    print("\n" + "=" * 80)
    print(f"3. SPRZĘŻENIE: Wpływ wyników Kroku 1 (wi*, fci* dla różnych lambda_i) na Krok 2 (wu*, dla w_cross={w_cross:.2f})")
    print("=" * 80)
    wu_grid = np.linspace(8.0, 25.0, 35)

    results = []
    for r in res_step1:
        lam = r["lam"]
        wi = r["wi_opt"]
        fci = r["fc_i_opt"]
        best_c = float("inf")
        best_wu = 15.0
        best_under, best_ise = 0.0, 0.0
        for wu in wu_grid:
            c, under, ise = eval_krok2(float(wu), wi, fci, w_cross)
            if c < best_c:
                best_c = c
                best_wu = float(wu)
                best_under, best_ise = under, ise
        results.append({
            "lam": lam,
            "wi_from_step1": wi,
            "fci_from_step1": fci,
            "wu_opt": best_wu,
            "cost": best_c,
            "undershoot": best_under,
            "ise": best_ise,
        })
        print(f"  lam={lam:4.2f} -> Krok 1: (wi*={wi:.4f}, fci*={fci:5.1f}Hz) --> Krok 2: wu* = {best_wu:4.1f} | zapad = {best_under:.2f} V, ISE = {best_ise:.4f}, J2 = {best_c:.4f}")
    return results


# ---------------------------------------------------------------------------
# Wykresy i zapis wynikow
# ---------------------------------------------------------------------------
def plot_sensitivity(res_step1: list[dict], res_step2: list[dict], res_cross: list[dict]):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    # Wykres 1: Wplyw lambda_i na wi* oraz fc_i* (Krok 1)
    lams = [r["lam"] for r in res_step1]
    wis = [r["wi_opt"] for r in res_step1]
    fcis = [r["fc_i_opt"] for r in res_step1]

    ax1 = axs[0]
    ax1_f = ax1.twinx()
    p1 = ax1.plot(lams, wis, "o-", color="tab:blue", lw=1.8, markersize=5, label=r"$w_i^*$")
    p2 = ax1_f.plot(lams, fcis, "s--", color="tab:red", lw=1.5, markersize=5, label=r"$f_{c,i}^*$ [Hz]")
    ax1.set_xlabel(r"Waga błędu prądu $\lambda_i$")
    ax1.set_ylabel(r"Optymalna waga $w_i^*$", color="tab:blue")
    ax1_f.set_ylabel(r"Optymalny filtr $f_{c,i}^*$ [Hz]", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1_f.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title(r"Krok 1: 2D PSO ($w_i^*, f_{c,i}^*$ vs $\lambda_i$)", fontsize=10, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Wykres 2: Wplyw w_cross na wu* (Krok 2)
    wcs = [r["w_cross"] for r in res_step2]
    wus = [r["wu_opt"] for r in res_step2]
    axs[1].plot(wcs, wus, "s-", color="tab:green", lw=1.8, markersize=6)
    axs[1].axhline(15.0, color="gray", ls="--", alpha=0.6, label="Punkt roboczy (15.0)")
    axs[1].set_xlabel(r"Waga kary za zapad $w_{\mathrm{cross}}$")
    axs[1].set_ylabel(r"Optymalna waga $w_u^*$")
    axs[1].set_title(r"Krok 2: Wrażliwość $w_u^*$ na $w_{\mathrm{cross}}$", fontsize=10, fontweight="bold")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(fontsize=8)

    # Wykres 3: Sprzezenie Krok 1 -> Krok 2
    lams_c = [r["lam"] for r in res_cross]
    wus_c = [r["wu_opt"] for r in res_cross]
    axs[2].plot(lams_c, wus_c, "^-", color="tab:purple", lw=1.8, markersize=6)
    axs[2].axhline(15.0, color="gray", ls="--", alpha=0.6, label="Punkt roboczy (15.0)")
    axs[2].set_xlabel(r"Waga $\lambda_i$ przyjęta w Kroku 1")
    axs[2].set_ylabel(r"Wyznaczona waga $w_u^*$ w Kroku 2")
    axs[2].set_title(r"Sprzężenie: Wynik Kroku 1 $\to$ Krok 2", fontsize=10, fontweight="bold")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(fontsize=8)

    fig.tight_layout()
    out_img = "optymalizacja/wykres_badanie_wrazliwosci.png"
    fig.savefig(out_img, dpi=140)
    plt.close(fig)
    print(f"\nZapisano wykres wrażliwości: {out_img}")


def generate_report_pdf(res_step1: list[dict], res_step2: list[dict], res_cross: list[dict], plot_img: str):
    filename = "raport_badanie_wrazliwosci_profesor.pdf"
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        topMargin=12 * mm,
        bottomMargin=14 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName=FONT_BOLD,
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=2,
    )

    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8.0,
        leading=11.0,
        textColor=colors.HexColor("#4A5568"),
    )

    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName=FONT_BOLD,
        fontSize=9.0,
        leading=11.5,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=3,
        spaceAfter=1.5,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor("#2D3748"),
        spaceAfter=2,
    )

    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName=FONT_NORMAL,
        fontSize=7.2,
        leading=8.8,
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
        fontName=FONT_NORMAL,
        alignment=0,
    )

    story = []

    # -----------------------------------------------------------------------
    # STRONA 1: Nagłówek, Wprowadzenie, Krok 1 (Stan ustalony)
    # -----------------------------------------------------------------------
    story.append(Paragraph("Raport: Badanie wrażliwości i odporności algorytmu Trend MF-BB", title_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceBefore=1, spaceAfter=4))

    meta_data = [
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Do:</font> Prof. dr hab. inż. Grzegorz Iwański", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Data:</font> Wrzesień 2026", meta_style),
        ],
        [
            Paragraph(f"<font name='{FONT_BOLD}'>Od:</font> inż. Rafał Kaczyński (album: 342208)", meta_style),
            Paragraph(f"<font name='{FONT_BOLD}'>Temat:</font> Badanie wrażliwości algorytmu dwutorowego (wrażliwość na wagi funkcji celu)", meta_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[90 * mm, 84 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ("PADDING", (0, 0), (-1, -1), 3),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph(f"<font name='{FONT_BOLD}'>1. Cel zestawienia</font>", h2_style))
    story.append(Paragraph(
        "Niniejszy raport zawiera zestawienie danych liczbowych i wykresów z badania wrażliwości dwutorowej procedury "
        "optymalizacyjnej regulatora Trend MF-BB. Zgodnie z wytycznymi zbadano, jak zmiana wag wskaźników jakości "
        "wpływa na wartości parametrów zwracane przez algorytm w obu krokach oraz jak wynik Kroku 1 wpływa na Krok 2.",
        body_style
    ))
    story.append(Spacer(1, 1.5 * mm))

    story.append(Paragraph(f"<font name='{FONT_BOLD}'>2. Krok 1 (Stan ustalony): Wpływ relacji błąd napięcia vs. błąd prądu (λ_i)</font>", h2_style))
    story.append(Paragraph(
        "W stanie ustalonym ($w_u = 0$, filtry $f_{c,i} = f_{c,u} = 321.0\\text{ Hz}$) funkcja celu ma postać:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>J_1(w_i) = IAE_u + λ_i · IAE_i</b><br/>"
        "gdzie $\\mathrm{IAE}_u = \\mathrm{mean}(|u_{\\mathrm{dc}} - 240\\text{ V}|)$ oraz $\\mathrm{IAE}_i = \\mathrm{mean}(|i_{\\mathrm{des}} - i_L|)$ "
        "są liczone w oknie stanu ustalonego pod obciążeniem 1.2 kW ($t \\in [120;\\, 150]\\text{ ms}$).<br/>"
        "W tabeli 1 zestawiono wartości $w_i^*$ zwrócone przez algorytm przy zmianie wagi $\\lambda_i$ w przedziale od 0.10 do 5.00:",
        body_style
    ))
    story.append(Spacer(1, 1 * mm))

    t1_headers = [
        Paragraph("Waga λ_i", table_cell_bold),
        Paragraph("Optymalne w_i*", table_cell_bold),
        Paragraph("Filtr f_c,i* [Hz]", table_cell_bold),
        Paragraph("Błąd napięcia<br/>IAE_u [V]", table_cell_bold),
        Paragraph("Błąd prądu<br/>IAE_i [A]", table_cell_bold),
        Paragraph("Tętnienia i_L<br/>p-p [A]", table_cell_bold),
        Paragraph("Koszt J_1", table_cell_bold),
    ]
    t1_data = [t1_headers]
    for r in res_step1:
        t1_data.append([
            Paragraph(f"{r['lam']:.2f}", table_cell),
            Paragraph(f"<b>{r['wi_opt']:.4f}</b>", table_cell),
            Paragraph(f"{r['fc_i_opt']:.1f}", table_cell),
            Paragraph(f"{r['iae_u']:.4f}", table_cell),
            Paragraph(f"{r['iae_i']:.3f}", table_cell),
            Paragraph(f"{r['iL_pp']:.2f}", table_cell),
            Paragraph(f"{r['cost']:.4f}", table_cell),
        ])

    t1_table = Table(t1_data, colWidths=[20 * mm, 28 * mm, 26 * mm, 25 * mm, 25 * mm, 25 * mm, 25 * mm])
    t1_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 2.5),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#EBF8FF")),
    ]))
    story.append(t1_table)
    story.append(Spacer(1, 2 * mm))

    # -----------------------------------------------------------------------
    # STRONA 2: Krok 2 (Stan przejściowy), Wpływ Kroku 1 na Krok 2
    # -----------------------------------------------------------------------
    story.append(PageBreak())

    story.append(Paragraph(f"<font name='{FONT_BOLD}'>3. Krok 2 (Stan przejściowy): Wzór funkcji celu i wpływ wagi kary (w_cross)</font>", h2_style))
    story.append(Paragraph(
        "Dla stanu przejściowego (skok napięcia $240 \\to 160\\text{ V}$ w $t = 150\\text{ ms}$, zamrożone nominalne parametry z Kroku 1) "
        "funkcja celu łączy całkę kwadratu błędu z karą za zapad poniżej 160 V:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>J_2(w_u) = ISE_trans + w_cross · (ΔU_undershoot)²</b><br/>"
        "gdzie:<br/>"
        "• <b>ISE_trans = ∫ [u_dc(t) - 160 V]² dt</b> liczona w oknie 150–165 ms,<br/>"
        "• <b>ΔU_undershoot = max(0, 160 V - min(u_dc))</b> liczona w oknie 150–155 ms.<br/>"
        "W tabeli 2 przedstawiono wartości $w_u^*$ zwrócone przez algorytm przy zmianie wagi $w_{\\mathrm{cross}}$ w przedziale od 0.00 do 5.00:",
        body_style
    ))
    story.append(Spacer(1, 1 * mm))

    t2_headers = [
        Paragraph("Waga w_cross", table_cell_bold),
        Paragraph("Zwrócone w_u*", table_cell_bold),
        Paragraph("Zapad napięcia [V]", table_cell_bold),
        Paragraph("Kwadrat błędu ISE", table_cell_bold),
        Paragraph("Wartość J_2", table_cell_bold),
    ]
    t2_data = [t2_headers]
    for r in res_step2:
        t2_data.append([
            Paragraph(f"{r['w_cross']:.2f}", table_cell),
            Paragraph(f"<b>{r['wu_opt']:.1f}</b>", table_cell),
            Paragraph(f"{r['undershoot']:.2f} V", table_cell),
            Paragraph(f"{r['ise']:.4f}", table_cell),
            Paragraph(f"{r['cost']:.4f}", table_cell),
        ])

    t2_table = Table(t2_data, colWidths=[32 * mm, 34 * mm, 36 * mm, 36 * mm, 36 * mm])
    t2_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 2.2),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#EBF8FF")),
    ]))
    story.append(t2_table)
    story.append(Spacer(1, 2.5 * mm))

    story.append(Paragraph(f"<font name='{FONT_BOLD}'>4. Wpływ wyników Kroku 1 na Krok 2 (Sprzężenie etapów)</font>", h2_style))
    story.append(Paragraph(
        "W tabeli 3 zestawiono wyniki Kroku 2 uzyskane dla par $(w_i^*, f_{c,i}^*)$ wyznaczonych "
        "w Kroku 1 dla poszczególnych wag $\\lambda_i$ (przy nominalnej wadze $w_{\\mathrm{cross}} = 0.10$):",
        body_style
    ))
    story.append(Spacer(1, 1 * mm))

    t3_headers = [
        Paragraph("λ_i (Krok 1)", table_cell_bold),
        Paragraph("w_i* (Krok 1)", table_cell_bold),
        Paragraph("f_c,i* [Hz] (Krok 1)", table_cell_bold),
        Paragraph("Wyznaczone w_u*", table_cell_bold),
        Paragraph("Zapad napięcia [V]", table_cell_bold),
        Paragraph("Koszt J_2", table_cell_bold),
    ]
    t3_data = [t3_headers]
    for r in res_cross:
        t3_data.append([
            Paragraph(f"{r['lam']:.2f}", table_cell),
            Paragraph(f"{r['wi_from_step1']:.4f}", table_cell),
            Paragraph(f"{r['fci_from_step1']:.1f}", table_cell),
            Paragraph(f"<b>{r['wu_opt']:.1f}</b>", table_cell),
            Paragraph(f"{r['undershoot']:.2f} V", table_cell),
            Paragraph(f"{r['cost']:.4f}", table_cell),
        ])

    t3_table = Table(t3_data, colWidths=[24 * mm, 30 * mm, 32 * mm, 30 * mm, 30 * mm, 28 * mm])
    t3_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 2.2),
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#EBF8FF")),
    ]))
    story.append(t3_table)
    story.append(Spacer(1, 3 * mm))

    sign_txt = (
        f"<font name='{FONT_ITALIC}'>Raport opracowany na podstawie symulacji numerycznych algorytmu dwutorowego.<br/>"
        f"Z poważaniem,<br/><font name='{FONT_BOLD}'>inż. Rafał Kaczyński</font> (album 342208)</font>"
    )
    story.append(Paragraph(sign_txt, body_style))

    # -----------------------------------------------------------------------
    # STRONA 3: Załącznik graficzny (Wykres 3-panelowy)
    # -----------------------------------------------------------------------
    story.append(PageBreak())

    story.append(Paragraph("Załącznik: Wykresy badania wrażliwości parametrów", h2_style))
    story.append(Paragraph(
        "Poniższy wykres trójpanelowy przedstawia zależności parametrów optymalnych od wag funkcji celu oraz wzajemne powiązanie kroków:",
        body_style
    ))
    story.append(Spacer(1, 2 * mm))

    if os.path.exists(plot_img):
        story.append(Image(plot_img, width=174 * mm, height=72 * mm))
        story.append(Spacer(1, 3 * mm))

    desc_w1 = (
        "<b>Wykres A (po lewej):</b> Wartości $w_i^*$ (oś lewa) oraz $f_{c,i}^*$ (oś prawa) wyznaczone przez 2D PSO w Kroku 1 "
        "w zależności od wagi błędu prądu $\\lambda_i$."
    )
    desc_w2 = (
        "<b>Wykres B (w środku):</b> Wartości $w_u^*$ wyznaczone w Kroku 2 w zależności od wagi kary za zapad $w_{\\mathrm{cross}}$."
    )
    desc_w3 = (
        "<b>Wykres C (po prawej):</b> Wartości $w_u^*$ wyznaczone w Kroku 2 na podstawie par $(w_i^*, f_{c,i}^*)$ przekazanych z Kroku 1 "
        "dla poszczególnych wag $\\lambda_i$."
    )
    story.append(Paragraph(desc_w1, body_style))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(desc_w2, body_style))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(desc_w3, body_style))

    def make_canvas(fname, **kw):
        return NumberedCanvas(fname, footer_title="Badanie wrażliwości dwutorowego algorytmu Trend MF-BB | PW", **kw)

    doc.build(story, canvasmaker=make_canvas)
    print(f"Wygenerowano raport PDF: {filename}")


def main():
    print("=" * 80)
    print("BADANIE WRAŻLIWOŚCI DWUTOROWEGO ALGORYTMU OPTYMALIZACYJNEGO (2D PSO KROK 1 -> KROK 2)")
    print("=" * 80)
    t0 = time.time()

    # Krok 1: 2D PSO dla (wi, fci) dla roznych lambda_i
    res1 = sweep_step1_sensitivity()

    # Wybieramy nominalna pare z Kroku 1 (dla lambda = 1.0)
    nom = [r for r in res1 if r["lam"] == 1.00][0]
    wi_nom, fci_nom = nom["wi_opt"], nom["fc_i_opt"]

    # Krok 2: Badanie wplywu w_cross dla nominalnej pary z Kroku 1
    res2 = sweep_step2_wcross_sensitivity(wi_nom, fci_nom)

    # Sprzezenie: Jak rzeczywiste wyniki z Kroku 1 przekladaja sie na Krok 2
    res_cross = sweep_step2_cross_influence(res1, w_cross=0.10)

    # Zapis danych numerycznych
    out_npz = "optymalizacja/badanie_wrazliwosci_results.npz"
    np.savez(
        out_npz,
        step1_lambdas=np.array([r["lam"] for r in res1]),
        step1_wi_opt=np.array([r["wi_opt"] for r in res1]),
        step1_fci_opt=np.array([r["fc_i_opt"] for r in res1]),
        step2_wcross=np.array([r["w_cross"] for r in res2]),
        step2_wu_opt=np.array([r["wu_opt"] for r in res2]),
        cross_lam=np.array([r["lam"] for r in res_cross]),
        cross_wi=np.array([r["wi_from_step1"] for r in res_cross]),
        cross_fci=np.array([r["fci_from_step1"] for r in res_cross]),
        cross_wu=np.array([r["wu_opt"] for r in res_cross]),
    )
    print(f"\nZapisano dane numeryczne: {out_npz}")

    plot_sensitivity(res1, res2, res_cross)
    plot_img = "optymalizacja/wykres_badanie_wrazliwosci.png"

    print("Generowanie formalnego raportu PDF dla Profesora...")
    generate_report_pdf(res1, res2, res_cross, plot_img)
    print(f"\nBadanie wrażliwości zakończone w {time.time()-t0:.1f} s.")


if __name__ == "__main__":
    main()
