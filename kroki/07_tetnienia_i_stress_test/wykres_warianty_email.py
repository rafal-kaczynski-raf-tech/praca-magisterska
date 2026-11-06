"""Zbiorczy wykres wynikow dla promotora (e-mail / Teams).

Trzy panele 'na rzut oka':
  A) slupki -- tetnienia PRADU (p-p) dla 4 wariantow, wariant 1 wyrozniony,
  B) slupki -- tetnienia NAPIECIA (p-p) dla 4 wariantow,
  C) przebieg prADU w oknie ustalonym: odniesienie (ITAE) vs wariant 1,
     pokazuje realne zmniejszenie falowania.

Uruchomienie:  python -m optymalizacja.wykres_warianty_email
Zapisuje:      wykres_warianty_email.png  (katalog glowny)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import default_config
from optymalizacja.analiza_ripple import simulate, analyse

# Kolejnosc i pliki nastaw (npz z PSO)
CASES = [
    ("Odniesienie\n(napiecie)",  "optymalizacja/pso_stress_itae.npz",          "#9e9e9e"),
    ("Wariant 1\n(oba uchyby)",   "optymalizacja/pso_stress_currentaware.npz",  "#2e7d32"),
    ("Wariant 2\n(oscylacje)",    "optymalizacja/pso_current_oscillation.npz",  "#1565c0"),
    ("Wariant 3\n(wielk. pradu)", "optymalizacja/pso_stress_currenteffort.npz", "#1565c0"),
]

ZOOM = (0.052, 0.060)     # okno do liczenia tetnien (stan ustalony 280 V)
WAVE = (0.0560, 0.0572)   # waskie okno (1.2 ms) do czytelnego przebiegu


def main() -> None:
    base = default_config()
    u_ref0 = base.controller.u_ref
    i_max = base.controller.i_max

    labels, colors = [], []
    iL_pp, vC_pp, iL_pct = [], [], []
    waveforms = {}   # label -> (t, iL) w oknie ZOOM

    for label, path, color in CASES:
        d = np.load(path)
        wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
        res = simulate(wi, fc)
        a = analyse(res, u_ref0, i_max)
        iL_mean = float(np.mean(res["i_L"][res["t"] > 0.05]))

        labels.append(label)
        colors.append(color)
        iL_pp.append(a["iL_ripple_pp"])
        vC_pp.append(a["v_ripple_pp"])
        iL_pct.append(100.0 * a["iL_ripple_pp"] / iL_mean)

        t = res["t"]
        m = (t >= WAVE[0]) & (t <= WAVE[1])
        waveforms[label.replace("\n", " ")] = (t[m] * 1e3, res["i_L"][m])

    x = np.arange(len(labels))

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.3], wspace=0.32)

    # --- Panel A: tetnienia pradu ---
    axA = fig.add_subplot(gs[0, 0])
    bars = axA.bar(x, iL_pp, color=colors, edgecolor="black", linewidth=0.6)
    axA.set_title("Tetnienia PRADU (p-p)", fontsize=12, fontweight="bold")
    axA.set_ylabel("amplituda tetnien [A]")
    axA.set_xticks(x)
    axA.set_xticklabels(labels, fontsize=8)
    axA.grid(axis="y", alpha=0.3)
    for xi, v, pct in zip(x, iL_pp, iL_pct):
        axA.text(xi, v + 0.12, f"{v:.2f} A\n({pct:.0f}%)", ha="center",
                 va="bottom", fontsize=8.5)
    axA.set_ylim(0, max(iL_pp) * 1.25)

    # --- Panel B: tetnienia napiecia ---
    axB = fig.add_subplot(gs[0, 1])
    axB.bar(x, vC_pp, color=colors, edgecolor="black", linewidth=0.6)
    axB.set_title("Tetnienia NAPIECIA (p-p)", fontsize=12, fontweight="bold")
    axB.set_ylabel("amplituda tetnien [V]")
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=8)
    axB.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, vC_pp):
        axB.text(xi, v + 0.02, f"{v:.2f} V", ha="center", va="bottom",
                 fontsize=8.5)
    axB.set_ylim(0, max(vC_pp) * 1.22)

    # --- Panel C: przebieg pradu odniesienie vs wariant 1 ---
    axC = fig.add_subplot(gs[0, 2])
    t0, iL0 = waveforms["Odniesienie (napiecie)"]
    t1, iL1 = waveforms["Wariant 1 (oba uchyby)"]
    axC.plot(t0, iL0, color="#9e9e9e", lw=1.3, label="Odniesienie (tylko napiecie)")
    axC.plot(t1, iL1, color="#2e7d32", lw=1.3, label="Wariant 1 (oba uchyby)")
    axC.set_title("Prad cewki -- stan ustalony (280 V)", fontsize=12,
                  fontweight="bold")
    axC.set_xlabel("czas [ms]")
    axC.set_ylabel("prad cewki $i_L$ [A]")
    axC.grid(alpha=0.3)
    axC.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Porownanie wariantow funkcji celu -- tetnienia pradu i napiecia "
                 "(scenariusz: 8 skokow 240<->280 V)",
                 fontsize=12.5, fontweight="bold", y=1.02)

    out = "wykres_warianty_email.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"ZAPISANO: {out}")
    for lbl, pp, pct, v in zip([l.replace(chr(10), ' ') for l in labels],
                               iL_pp, iL_pct, vC_pp):
        print(f"  {lbl:32s} iL_pp={pp:.2f}A ({pct:.0f}%)  vC_pp={v:.2f}V")


if __name__ == "__main__":
    main()
