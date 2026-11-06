"""Porownanie wariantow funkcji celu uwzgledniajacych prad (uwagi prof. Iwanskiego).

Zestawia 4 strojenia sterownika MF-BB na scenariuszu trudnym:
  - ITAE             - bazowa funkcja (tylko napiecie, wazona czasem)
  - Wariant 1        - CurrentAware: IAE_u + lambda * IAE(i_des - i_L)
  - Wariant 2        - CurrentOscillation: IAE_u + mu * rozstep(4 probek)^2
  - Composite (zly)  - przyklad rozwiazania z duzym tetnieniem pradu

Generuje porownanie_warianty_prad.png (napiecie + prad, zoom na skok ref i ust.).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.scenarios import hard_scenario
from optymalizacja.pso import T_END, _build_u_ref_arr

SCN = hard_scenario()

# (etykieta, wi, fc, kolor)
RUNS = [
    ("ITAE (baza)",        0.276,  9493.0, "tab:gray"),
    ("Wariant 1 (uchyb i)", 0.3613, 9187.6, "tab:blue"),
    ("Wariant 2 (oscyl. 4 prob.)", 0.3639, 9050.6, "tab:green"),
    ("Composite (zly)",    0.136,   618.0, "tab:red"),
]


def simulate(wi, fc):
    base = default_config()
    cfg = replace(base, controller=replace(base.controller, wi=wi, fc_lpf=fc),
                  scenario=SCN, T_end=T_END)
    return Simulator(cfg).run()


def main():
    results = [(lab, simulate(wi, fc), col) for lab, wi, fc, col in RUNS]
    t0 = results[0][1]["t"]
    u_ref = _build_u_ref_arr(t0, default_config().controller.u_ref, SCN)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex="col")

    # Okna czasowe
    t_ref = SCN.ref_step_time          # 0.10 s - skok referencji 240->260 V
    zoom_tr = (t_ref - 0.002, t_ref + 0.008)   # przejscie
    zoom_ss = (0.18, 0.182)                     # stan ustalony (2 ms)

    for col, (lo, hi), title in [(0, zoom_tr, "Skok referencji 240->260 V (stan przejsciowy)"),
                                  (1, zoom_ss, "Stan ustalony (zoom 2 ms)")]:
        # referencja napiecia
        mref = (t0 >= lo) & (t0 <= hi)
        ax[0, col].plot(t0[mref] * 1e3, u_ref[mref], "k--", lw=1, alpha=0.6,
                        label="u_ref")
        for lab, res, color in results:
            t = res["t"]; m = (t >= lo) & (t <= hi)
            ax[0, col].plot(t[m] * 1e3, res["v_C"][m], color=color, lw=1.2, label=lab)
            ax[1, col].plot(t[m] * 1e3, res["i_L"][m], color=color, lw=1.2, label=lab)
        ax[0, col].set_title(title, fontsize=10)
        ax[0, col].set_ylabel("u_dc [V]")
        ax[1, col].set_ylabel("i_L [A]")
        ax[1, col].set_xlabel("czas [ms]")
        for r in (0, 1):
            ax[r, col].grid(True, alpha=0.3)

    ax[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Porownanie funkcji celu uwzgledniajacych prad - sterownik MF-BB",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = "porownanie_warianty_prad.png"
    fig.savefig(out, dpi=130)
    print("Zapisano:", out)


if __name__ == "__main__":
    main()
