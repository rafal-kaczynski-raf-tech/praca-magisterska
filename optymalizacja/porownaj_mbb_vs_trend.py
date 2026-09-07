"""Porownanie: Poprzedni zwyciezca (Model-Based BB z korekta pradowa, IEEE JESTIE)
vs Nowa technika (Trend MF-BB z filtrowanym trendem napiecia, nota 2026).

Dokladnie ten sam scenariusz dynamiczny prof. Iwanskiego (0.30s):
- V_in = 100V, L = 750 uH, C = 200 uF, R_load = 47.619 Ohm (1.2 kW @ 240V)
- Odciecia obciazenia w t=0.05, 0.10, 0.20, 0.25 s
- Skok referencji 240V -> 160V w t=0.15 s

Uruchomienie:  python -m optymalizacja.porownaj_mbb_vs_trend
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optymalizacja.analiza_scenariusz_prof import simulate as sim_mbb, analyse as analyse_prof
from optymalizacja.pso_trend_mfbb import simulate_with_params as sim_trend
from optymalizacja.scenarios import professor_scenario, PROF_T_END


def main():
    scn = professor_scenario()
    u_ref0 = 240.0

    print("=" * 85)
    print("POROWNANIE METOD W SCENARIUSZU PROFESORA (0.30s, C=200uF, L=750uH)")
    print("=" * 85)

    # 1. Poprzedni zwyciezca: Model-Based Bang-Bang (wi=0.2631, fc=7056 Hz, current_correction=True)
    res_mbb = sim_mbb(wi=0.263079, fc=7056.015)
    rows_mbb = analyse_prof(res_mbb, u_ref0, scn.ref_step_time, scn.ref_step_value)

    # 2. Nowa technika: Trend MF-BB zoptymalizowany przez PSO dla C=200uF
    # Nastawy z PSO: wi=0.3384, wv=0.7194, fc_i=922.9 Hz, fc_u=235.1 Hz
    res_trend = sim_trend(wi=0.3384, wv=0.7194, fc_i=922.9, fc_u=235.1, C_val=200e-6)
    rows_trend = analyse_prof(res_trend, u_ref0, scn.ref_step_time, scn.ref_step_value)

    # 3. Klasyczny MF-BB (bez trendu, wv=0.0)
    res_base_mfbb = sim_trend(wi=0.3384, wv=0.0, fc_i=922.9, fc_u=235.1, C_val=200e-6)
    rows_base = analyse_prof(res_base_mfbb, u_ref0, scn.ref_step_time, scn.ref_step_value)

    # Wypisanie tabeli
    print(f"\n{'Segment czasowy':<22} | {'Model-Based BB (IEEE)':<22} | {'Nowy Trend MF-BB (PSO)':<22} | {'Klasyczny MF-BB (wv=0)':<20}")
    print("-" * 95)

    # Segment [0.10, 0.15) s -- 240V ON
    r_m = rows_mbb[1]
    r_t = rows_trend[1]
    r_b = rows_base[1]
    print(f"[0.10, 0.15)s (240V, ON)")
    print(f"  v_err [V]            | {r_m['v_err_V']:+8.3f} V             | {r_t['v_err_V']:+8.3f} V             | {r_b['v_err_V']:+8.3f} V")
    print(f"  iL_mean [A]          | {r_m['iL_mean_A']:8.2f} A             | {r_t['iL_mean_A']:8.2f} A             | {r_b['iL_mean_A']:8.2f} A")
    print(f"  iL_pp [A]            | {r_m['iL_ripple_pp_A']:8.3f} A             | {r_t['iL_ripple_pp_A']:8.3f} A             | {r_b['iL_ripple_pp_A']:8.3f} A")
    print(f"  iL_ripple [%]        | {r_m['iL_ripple_pct']:8.2f} %             | {r_t['iL_ripple_pct']:8.2f} %             | {r_b['iL_ripple_pct']:8.2f} %")

    # Segment [0.25, 0.30) s -- 160V ON
    r_m160 = rows_mbb[4]
    r_t160 = rows_trend[4]
    r_b160 = rows_base[4]
    print(f"\n[0.25, 0.30)s (160V, ON)")
    print(f"  v_err [V]            | {r_m160['v_err_V']:+8.3f} V             | {r_t160['v_err_V']:+8.3f} V             | {r_b160['v_err_V']:+8.3f} V")
    print(f"  iL_mean [A]          | {r_m160['iL_mean_A']:8.2f} A             | {r_t160['iL_mean_A']:8.2f} A             | {r_b160['iL_mean_A']:8.2f} A")
    print(f"  iL_pp [A]            | {r_m160['iL_ripple_pp_A']:8.3f} A             | {r_t160['iL_ripple_pp_A']:8.3f} A             | {r_b160['iL_ripple_pp_A']:8.3f} A")
    print(f"  iL_ripple [%]        | {r_m160['iL_ripple_pct']:8.2f} %             | {r_t160['iL_ripple_pct']:8.2f} %             | {r_b160['iL_ripple_pct']:8.2f} %")

    print(f"\nCaly horyzont:")
    print(f"  iL_peak [A]          | {np.max(res_mbb['i_L']):8.2f} A             | {np.max(res_trend['i_L']):8.2f} A             | {np.max(res_base_mfbb['i_L']):8.2f} A")

    # Wykres porownawczy
    t = res_mbb["t"] * 1e3
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax[0].plot(t, res_mbb["v_C"], label="Poprzedni zwyciezca: Model-Based BB (Eq.14-17)", color="tab:blue", lw=0.9)
    ax[0].plot(t, res_trend["v_C"], label="Nowa technika: Trend MF-BB (z trendem delta_uf)", color="tab:orange", lw=0.9, alpha=0.9)
    ax[0].plot(t, res_base_mfbb["v_C"], label="Klasyczny MF-BB (bez trendu, wv=0)", color="tab:green", lw=0.8, ls="--", alpha=0.7)
    ax[0].set_ylabel("u_dc [V]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="lower right")
    ax[0].set_title("Porownanie metod sterowania w scenariuszu dynamicznym prof. Iwanskiego")

    ax[1].plot(t, res_mbb["i_L"], label="Model-Based BB", color="tab:blue", lw=0.8)
    ax[1].plot(t, res_trend["i_L"], label="Trend MF-BB (z trendem)", color="tab:orange", lw=0.8, alpha=0.9)
    ax[1].plot(t, res_base_mfbb["i_L"], label="Klasyczny MF-BB", color="tab:green", lw=0.7, ls="--", alpha=0.7)
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_xlabel("czas [ms]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right")

    # Linie zdarzen
    for t_ev, lbl in [(50, "R->10M"), (100, "R->47.6"), (150, "ref 240->160V"), (200, "R->10M"), (250, "R->47.6")]:
        ax[0].axvline(t_ev, color="red", lw=0.7, ls=":", alpha=0.6)
        ax[1].axvline(t_ev, color="red", lw=0.7, ls=":", alpha=0.6)

    fig.tight_layout()
    out_png = "optymalizacja/porownanie_mbb_vs_trend.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nZapisano wykres porownawczy: {out_png}")


if __name__ == "__main__":
    main()
