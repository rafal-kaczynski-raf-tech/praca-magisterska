"""Generuje PDF z surowymi probkami t/v_C/i_L (okno zoom, 1.2 ms) dla kazdego
z 4 wariantow funkcji celu -- zalacznik do maila do promotora, zamiast tabel
wklejonych bezposrednio w tresc wiadomosci.

Uruchomienie:  python -m optymalizacja.tabela_probek_pdf
"""
from __future__ import annotations
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

from optymalizacja.analiza_ripple import simulate

CASES = [
    ("ITAE (odniesienie)", "optymalizacja/pso_stress_itae.npz"),
    ("Wariant 1 (oba uchyby)", "optymalizacja/pso_stress_currentaware.npz"),
    ("Wariant 2 (oscylacje 4 probek)", "optymalizacja/pso_current_oscillation.npz"),
    ("Wariant 3 (wielkosc pradu)", "optymalizacja/pso_stress_currenteffort.npz"),
]

ZOOM_WIN = (0.0560, 0.0572)   # 1.2 ms, ten sam zakres co przebiegi_wariantow_zoom.png
STEP = 20e-6                   # co 20 us -> ok. 60 wierszy

OUT_PATH = "przebiegi_probki_wariantow.pdf"

DIACRITICS = str.maketrans("ąęółśżźćń", "aeolszzcn")


def pl(text: str) -> str:
    return text.translate(DIACRITICS)


def build_rows(t: np.ndarray, v: np.ndarray, iL: np.ndarray) -> list[list[str]]:
    rows = [["t [ms]", "v_C [V]", "i_L [A]"]]
    tt = ZOOM_WIN[0]
    while tt <= ZOOM_WIN[1]:
        idx = int(np.argmin(np.abs(t - tt)))
        rows.append([f"{t[idx]*1e3:.4f}", f"{v[idx]:.3f}", f"{iL[idx]:.3f}"])
        tt += STEP
    return rows


def main() -> None:
    doc = SimpleDocTemplate(OUT_PATH, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=18 * mm,
                             leftMargin=20 * mm, rightMargin=20 * mm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(pl("Surowe probki napiecia i pradu -- okno stanu ustalonego (280 V)"),
                            styles["Title"]))
    story.append(Paragraph(pl(f"Zakres: {ZOOM_WIN[0]*1e3:.2f}-{ZOOM_WIN[1]*1e3:.2f} ms "
                               f"(to samo okno co przebiegi_wariantow_zoom.png), krok {STEP*1e6:.0f} us."),
                            styles["Normal"]))
    story.append(Spacer(1, 10 * mm))

    for i, (label, path) in enumerate(CASES):
        d = np.load(path)
        wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
        res = simulate(wi, fc)
        rows = build_rows(res["t"], res["v_C"], res["i_L"])

        story.append(Paragraph(pl(f"{label} -- w_i = {wi:.3f}, f_c = {fc:.0f} Hz"),
                                styles["Heading2"]))
        table = Table(rows, colWidths=[35 * mm, 35 * mm, 35 * mm], repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(table)
        if i < len(CASES) - 1:
            story.append(PageBreak())

    doc.build(story)
    print(f"Zapisano: {OUT_PATH}")


if __name__ == "__main__":
    main()
