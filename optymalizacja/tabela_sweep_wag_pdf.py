"""Generuje PDF z wynikami sweepu recznie dobranych wag czlonu pradowego
(lam/mu/gamma) dla 3 wariantow funkcji celu -- scenariusz stress (8 skokow
referencji 240<->280 V co 20 ms). Dla kazdego wariantu: tabela tetnien
i napiecia dla coraz MNIEJSZYCH (10/30/50% bazy) i coraz WIEKSZYCH
(110/130/150/200/300% bazy) wag, plus krotkie wnioski.

Dane wpisane na sztywno -- pochodza z faktycznych przebiegow PSO wykonanych
w tej sesji (pso_lambda_sweep.py, pso_lambda_sweep_up.py,
pso_weight_sweep_variants23.py, pso_weight_sweep_variants23_up.py).

Uruchomienie:  python -m optymalizacja.tabela_sweep_wag_pdf
"""
from __future__ import annotations
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

OUT_PATH = "sweep_wag_pradu_stress.pdf"

DIACRITICS = str.maketrans("ąęółśżźćńĄĘÓŁŚŻŹĆŃ", "aeolszzcnAEOLSZZCN")


def pl(text: str) -> str:
    return text.translate(DIACRITICS)


# (factor, waga, wi*, fc* [Hz], IAE_u [V*s], iL_pp [A], vC_pp [V], iL_peak [A])
WARIANT_1 = {
    "name": "Wariant 1 -- CurrentAware (uchyb pradu, waga lam, baza lam=1.0)",
    "rows": [
        (0.10, 0.10000, 0.3324, 2201.8, 0.6519, 8.263, 1.7271, 22.64),
        (0.30, 0.30000, 0.3643, 7694.1, 0.6536, 7.715, 1.6135, 22.63),
        (0.50, 0.50000, 0.5370, 6372.8, 0.6622, 7.149, 1.4877, 22.64),
        (1.00, 1.00000, 0.6383, 6370.7, 0.6739, 6.807, 1.3966, 22.64),
        (1.10, 1.10000, 0.6383, 5603.6, 0.6750, 6.694, 1.3630, 22.64),
        (1.30, 1.30000, 0.6564, 6907.0, 0.6789, 6.683, 1.3613, 22.63),
        (1.50, 1.50000, 0.6443, 6866.4, 0.6762, 6.798, 1.3945, 22.64),
        (2.00, 2.00000, 0.6566, 6604.3, 0.6789, 6.683, 1.3613, 22.63),
        (3.00, 3.00000, 0.6567, 6828.2, 0.6789, 6.683, 1.3613, 22.63),
    ],
}

WARIANT_2 = {
    "name": "Wariant 2 -- CurrentOscillation (rozstep w oknie 4 probek, waga mu, baza mu=0.02)",
    "rows": [
        (0.10, 0.00200, 0.3470, 6331.6, 0.6526, 7.857, 1.6348, 22.63),
        (0.30, 0.00600, 0.3324, 2201.8, 0.6519, 8.263, 1.7271, 22.64),
        (0.50, 0.01000, 0.4869, 4249.5, 0.6573, 7.382, 1.5990, 22.64),
        (1.00, 0.02000, 0.5373, 5942.8, 0.6623, 7.151, 1.4900, 22.63),
        (1.10, 0.02200, 0.5371, 5493.9, 0.6623, 7.151, 1.4901, 22.64),
        (1.30, 0.02600, 0.5326, 6128.3, 0.6644, 7.179, 1.5113, 22.64),
        (1.50, 0.03000, 0.6570, 6698.8, 0.6789, 6.683, 1.3613, 22.63),
        (2.00, 0.04000, 0.6564, 6757.8, 0.6789, 6.683, 1.3613, 22.63),
        (3.00, 0.06000, 0.6578, 7499.7, 0.6790, 6.690, 1.3682, 22.63),
    ],
}

WARIANT_3 = {
    "name": "Wariant 3 -- CurrentEffort (magnituda pradu, waga gamma, baza gamma=0.02)",
    "rows": [
        (0.10, 0.00200, 0.3447, 4482.0, 0.6518, 7.893, 1.6466, 22.63),
        (0.30, 0.00600, 0.3517, 1551.3, 0.6532, 8.039, 1.6757, 22.64),
        (0.50, 0.01000, 0.3324, 2201.8, 0.6519, 8.263, 1.7271, 22.64),
        (1.00, 0.02000, 0.3690, 9364.6, 0.6536, 7.682, 1.6128, 22.64),
        (1.10, 0.02200, 0.4284, 8741.3, 0.6560, 7.529, 1.5843, 22.64),
        (1.30, 0.02600, 0.5216, 7137.2, 0.6603, 7.270, 1.5495, 22.64),
        (1.50, 0.03000, 0.5155, 5325.2, 0.6602, 7.380, 1.5956, 22.64),
        (2.00, 0.04000, 0.5560, 9879.3, 0.6637, 7.279, 1.5484, 22.63),
        (3.00, 0.06000, 0.5728, 700.6, 0.6679, 7.163, 1.4926, 22.63),
    ],
}

VARIANTS = [WARIANT_1, WARIANT_2, WARIANT_3]

CONCLUSIONS = [
    "Tetnienia pradu (iL_pp) maleja wraz ze wzrostem wagi we wszystkich 3 "
    "wariantach, ale maja SUFIT: dla Wariantu 1 i 2 dalsze zwiekszanie wagi "
    "powyzej ok. 130-150% bazy juz nic nie zmienia (nastawy wi*, fc* i "
    "tetnienia zbiegaja do tych samych wartosci, ~6.68 A p-p).",
    "Wariant 3 zbliza sie do tego samego sufitu wolniej -- w testowanym "
    "zakresie (do 300% bazy) osiaga 7.16 A p-p, wciaz powyzej sufitu "
    "Wariantu 1/2.",
    "Koszt redukcji tetnien to wzrost bledu napiecia (IAE_u rosnie o ok. "
    "4% miedzy najnizsza a najwyzsza testowana waga we wszystkich "
    "wariantach) oraz wzrost tetnien napiecia przy NAJNIZSZYCH wagach "
    "(mniejsza waga = wieksze tetnienia i pradu, i napiecia rownoczesnie).",
    "Szczyt pradu (iL_peak) podczas skoku referencji pozostaje praktycznie "
    "staly (~22.6 A) niezaleznie od wagi, we wszystkich wariantach -- nie "
    "jest to wielkosc, na ktora waga funkcji celu ma wplyw.",
]


def build_table(rows: list[tuple]) -> Table:
    header = ["factor", "waga", "wi*", "fc* [Hz]", "IAE_u [V*s]",
              "iL_pp [A]", "vC_pp [V]", "iL_peak [A]"]
    data = [header]
    for factor, w, wi, fc, iae_u, il_pp, vc_pp, il_peak in rows:
        data.append([
            f"{factor:.2f}", f"{w:.5f}", f"{wi:.4f}", f"{fc:.1f}",
            f"{iae_u:.4f}", f"{il_pp:.3f}", f"{vc_pp:.4f}", f"{il_peak:.2f}",
        ])
    table = Table(data, colWidths=[16 * mm, 20 * mm, 18 * mm, 20 * mm,
                                    22 * mm, 20 * mm, 20 * mm, 22 * mm],
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        # wyroznienie wiersza bazowego (factor = 1.00)
        ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#fff2cc")),
    ]))
    return table


def main() -> None:
    doc = SimpleDocTemplate(OUT_PATH, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=18 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(pl("Sweep recznie dobranych wag czlonu pradowego -- "
                               "tetnienia i napiecie"), styles["Title"]))
    story.append(Paragraph(pl(
        "Scenariusz: stress-test (8 skokow referencji 240<->280 V co 20 ms). "
        "Dla kazdego z 3 wariantow funkcji celu (uwagi prof. Iwanskiego) "
        "reczna waga czlonu pradowego byla skalowana wzgledem wlasnej "
        "wartosci bazowej: 10% / 30% / 50% (coraz MNIEJSZE) oraz "
        "100% (baza, wyrozniona) / 110% / 130% / 150% / 200% / 300% "
        "(coraz WIEKSZE). Dla kazdej wartosci wagi uruchamiane bylo pelne "
        "PSO (20 czastek, 40 iteracji), a nastepnie symulacja stress z "
        "optymalnymi nastawami (wi*, fc*). Tetnienia (iL_pp, vC_pp) liczone "
        "jako peak-to-peak w ostatnich 40% kazdego z 8 segmentow miedzy "
        "skokami (usrednione), iL_peak = globalne maksimum pradu cewki."),
        styles["Normal"]))
    story.append(Spacer(1, 8 * mm))

    for variant in VARIANTS:
        story.append(Paragraph(pl(variant["name"]), styles["Heading2"]))
        story.append(build_table(variant["rows"]))
        story.append(Spacer(1, 8 * mm))

    story.append(Paragraph(pl("Wnioski"), styles["Heading2"]))
    for c in CONCLUSIONS:
        story.append(Paragraph(pl(f"- {c}"), styles["Normal"]))
        story.append(Spacer(1, 3 * mm))

    doc.build(story)
    print(f"Zapisano: {OUT_PATH}")


if __name__ == "__main__":
    main()
