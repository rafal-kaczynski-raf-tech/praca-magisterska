"""Weryfikacja nowego regulatora TrendMFBBController (notatka techniczna prof. Iwanskiego).

Porownanie wariantu bez trendu (wv = 0.0) oraz z trendem napiecia (wv > 0).
Scenariusz: skok referencji 200V -> 240V oraz skok obciazenia.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ConverterParams, TrendMFBBParams, SimulationConfig
from src.simulator import Simulator
from optymalizacja.scenarios import professor_scenario, PROF_T_END


def run_trend_sim(wv: float, wi: float = 0.2, fc_i: float = 321.0, fc_u: float = 321.0) -> dict:
    conv = ConverterParams(
        V_in=100.0,
        L=0.75e-3,       # 750 uH jak w Tabela 1 notatki technicznej
        C=1500e-6,       # 1500 uF jak w Tabela 1
        R_load=50.0,
        v_C0=100.0,
        i_L0=0.0,
    )
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,
        wi=wi,
        wv=wv,
        fc_i=fc_i,
        fc_u=fc_u,
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


def main():
    print("Uruchamiam symulacje bez trendu (wv = 0.0)...")
    res_no_trend = run_trend_sim(wv=0.0, wi=0.2)

    print("Uruchamiam symulacje z trendem (wv = 0.05)...")
    res_trend_005 = run_trend_sim(wv=0.05, wi=0.2)

    print("Uruchamiam symulacje z trendem (wv = 0.2)...")
    res_trend_02 = run_trend_sim(wv=0.2, wi=0.2)

    t = res_no_trend["t"] * 1e3

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t, res_no_trend["v_C"], label="MF-BB bazowy (wv=0.0)", lw=0.9)
    ax[0].plot(t, res_trend_005["v_C"], label="MF-BB z trendem (wv=0.05)", lw=0.9, alpha=0.85)
    ax[0].plot(t, res_trend_02["v_C"], label="MF-BB z trendem (wv=0.2)", lw=0.9, alpha=0.85)
    ax[0].set_ylabel("u_dc [V]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()
    ax[0].set_title("Porownanie MF-BB z filtrowanym trendem napiecia (notatka techniczna)")

    ax[1].plot(t, res_no_trend["i_L"], label="MF-BB bazowy (wv=0.0)", lw=0.8)
    ax[1].plot(t, res_trend_005["i_L"], label="MF-BB z trendem (wv=0.05)", lw=0.8, alpha=0.85)
    ax[1].plot(t, res_trend_02["i_L"], label="MF-BB z trendem (wv=0.2)", lw=0.8, alpha=0.85)
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_xlabel("czas [ms]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    out_png = "optymalizacja/wykres_trend_mfbb_test.png"
    fig.savefig(out_png, dpi=130)
    print(f"Zapisano wykres: {out_png}")


if __name__ == "__main__":
    main()
