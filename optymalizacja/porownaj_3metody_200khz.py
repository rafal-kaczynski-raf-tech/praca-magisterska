"""Porownanie trzech metod sterowania przy podwojonej czestotliwosci probkowania (200 kHz, Ts = 5 us):
1. Klasyczny MF-BB (bez trendu, bez modelu)
2. Model-Based BB (z predykcja i_L(k+1) i korekta modelowa i_delta, IEEE JESTIE)
3. Nowy Trend MF-BB (z filtrowanym trendem napiecia delta_u_f, nota techniczna 2026)

Pelny scenariusz dynamiczny prof. Iwanskiego (0.30s, C=200uF, L=750uH):
- 240V -> skoki obciazenia R (OFF @0.05s, ON @0.10s)
- Skok referencji 240V -> 160V @0.15s
- 160V -> skoki obciazenia R (OFF @0.20s, ON @0.25s)

Uruchomienie:  python -m optymalizacja.porownaj_3metody_200khz
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optymalizacja.analiza_scenariusz_prof import analyse as analyse_prof
from optymalizacja.scenarios import professor_scenario, PROF_T_END
from src.config import ConverterParams, ControllerParams, TrendMFBBParams, SimulationConfig
from src.simulator import Simulator


def main():
    scn = professor_scenario()
    u_ref0 = 240.0
    conv = ConverterParams(
        V_in=100.0,
        L=0.75e-3,
        C=200e-6,
        R_load=47.619,
        v_C0=100.0,
        i_L0=0.0,
    )
    ts_200k = 5e-6  # Ts = 5 us -> fs = 200 kHz

    print("=" * 105)
    print("SYMULACJA I POROWNANIE 3 METOD PRZY 200 kHz (Ts = 5 us) DLA SCENARIUSZA PROFESORA (0.30s)")
    print("=" * 105)

    # 1. Klasyczny MF-BB @ 200 kHz
    ctrl_mfbb = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=ts_200k,
        wi=0.3384,
        wv=0.0,
        fc_i=922.9,
        fc_u=235.1,
        i_max=20.0,
        i_min=-20.0,
    )
    res_mfbb = Simulator(SimulationConfig(converter=conv, controller=ctrl_mfbb, dt_phys=0.5e-6, T_end=PROF_T_END, scenario=scn)).run()
    rows_mfbb = analyse_prof(res_mfbb, u_ref0, scn.ref_step_time, scn.ref_step_value)

    # 2. Model-Based BB @ 200 kHz
    ctrl_mbb = ControllerParams(
        u_ref=240.0,
        T_s_ctrl=ts_200k,
        wi=0.2631,
        fc_lpf=7056.0,
        current_correction=True,
    )
    res_mbb = Simulator(SimulationConfig(converter=conv, controller=ctrl_mbb, dt_phys=0.5e-6, T_end=PROF_T_END, scenario=scn)).run()
    rows_mbb = analyse_prof(res_mbb, u_ref0, scn.ref_step_time, scn.ref_step_value)

    # 3. Nowy Trend MF-BB @ 200 kHz
    ctrl_trend = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=ts_200k,
        wi=0.3384,
        wv=0.7194,
        fc_i=922.9,
        fc_u=235.1,
        i_max=20.0,
        i_min=-20.0,
    )
    res_trend = Simulator(SimulationConfig(converter=conv, controller=ctrl_trend, dt_phys=0.5e-6, T_end=PROF_T_END, scenario=scn)).run()
    rows_trend = analyse_prof(res_trend, u_ref0, scn.ref_step_time, scn.ref_step_value)

    # Tabela zbiorcza wszystkich segmentow
    print(f"\n{'Segment czasowy':<22} | {'Wskaznik':<10} | {'1. Klasyczny MF-BB':<20} | {'2. Model-Based BB':<20} | {'3. Nowy Trend MF-BB':<20}")
    print("-" * 105)

    for i in range(len(rows_mfbb)):
        seg = rows_mfbb[i]["seg"]
        uref = rows_mfbb[i]["u_ref_V"]
        desc = f"{seg} ({uref:.0f}V)"

        v1 = f"{rows_mfbb[i]['v_err_V']:+7.3f} V"
        v2 = f"{rows_mbb[i]['v_err_V']:+7.3f} V"
        v3 = f"{rows_trend[i]['v_err_V']:+7.3f} V"

        rip1_pct = rows_mfbb[i]['iL_ripple_pct']
        rip2_pct = rows_mbb[i]['iL_ripple_pct']
        rip3_pct = rows_trend[i]['iL_ripple_pct']

        r1 = f"{rows_mfbb[i]['iL_ripple_pp_A']:6.3f} A ({rip1_pct:5.1f}%)" if not np.isnan(rip1_pct) else f"{rows_mfbb[i]['iL_ripple_pp_A']:6.3f} A (N/A)"
        r2 = f"{rows_mbb[i]['iL_ripple_pp_A']:6.3f} A ({rip2_pct:5.1f}%)" if not np.isnan(rip2_pct) else f"{rows_mbb[i]['iL_ripple_pp_A']:6.3f} A (N/A)"
        r3 = f"{rows_trend[i]['iL_ripple_pp_A']:6.3f} A ({rip3_pct:5.1f}%)" if not np.isnan(rip3_pct) else f"{rows_trend[i]['iL_ripple_pp_A']:6.3f} A (N/A)"

        print(f"{desc:<22} | {'v_err':<10} | {v1:<20} | {v2:<20} | {v3:<20}")
        print(f"{'':<22} | {'iL_ripple':<10} | {r1:<20} | {r2:<20} | {r3:<20}")
        print("-" * 105)

    # Statystyki calego horyzontu
    t = res_mfbb["t"]
    print(f"\nMETRYKI CALOSCIOWE (0 - 300 ms):")
    p1 = f"{np.max(res_mfbb['i_L']):.2f} A"
    p2 = f"{np.max(res_mbb['i_L']):.2f} A"
    p3 = f"{np.max(res_trend['i_L']):.2f} A"
    print(f"{'Maksymalny prad szczytowy (OCP 20A)':<35}: MF-BB={p1}, MB-BB={p2}, Trend={p3}")

    # Rysowanie wykresu porownawczego 3 metod przy 200 kHz
    tm = t * 1e3
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    ax[0].plot(tm, res_mbb["v_C"], label="2. Model-Based BB (IEEE, current_correction=True)", color="tab:blue", lw=0.9, alpha=0.85)
    ax[0].plot(tm, res_mfbb["v_C"], label="1. Klasyczny MF-BB (wv=0)", color="tab:green", lw=0.8, ls="--", alpha=0.7)
    ax[0].plot(tm, res_trend["v_C"], label="3. Nowy Trend MF-BB (wv=0.7194)", color="tab:red", lw=0.9, alpha=0.95)
    ax[0].set_ylabel("u_dc [V]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="lower right")
    ax[0].set_title("Porownanie 3 metod sterowania przy probkowaniu 200 kHz (Ts = 5 us) w pelnym scenariuszu")

    ax[1].plot(tm, res_mbb["i_L"], label="2. Model-Based BB", color="tab:blue", lw=0.7, alpha=0.85)
    ax[1].plot(tm, res_mfbb["i_L"], label="1. Klasyczny MF-BB", color="tab:green", lw=0.7, ls="--", alpha=0.7)
    ax[1].plot(tm, res_trend["i_L"], label="3. Nowy Trend MF-BB", color="tab:red", lw=0.7, alpha=0.95)
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_xlabel("czas [ms]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right")

    for t_ev, lbl in [(50, "R->10M"), (100, "R->47.6"), (150, "ref 240->160V"), (200, "R->10M"), (250, "R->47.6")]:
        ax[0].axvline(t_ev, color="k", lw=0.7, ls=":", alpha=0.5)
        ax[1].axvline(t_ev, color="k", lw=0.7, ls=":", alpha=0.5)

    fig.tight_layout()
    out_png_pelny = "optymalizacja/porownanie_3metody_200khz_pelny.png"
    fig.savefig(out_png_pelny, dpi=130)
    print(f"Zapisano wykres pelny: {out_png_pelny}")

    # Rysowanie wykresu ZOOM 140 - 210 ms (skok referencji @150ms + wylaczenie obciazenia @200ms)
    mask_zoom = (t >= 0.140) & (t <= 0.210)
    tm_z = t[mask_zoom] * 1e3

    fig_z, ax_z = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    ax_z[0].plot(tm_z, res_mbb["v_C"][mask_zoom], label="2. Model-Based BB (IEEE)", color="tab:blue", lw=1.0, alpha=0.85)
    ax_z[0].plot(tm_z, res_mfbb["v_C"][mask_zoom], label="1. Klasyczny MF-BB", color="tab:green", lw=0.9, ls="--", alpha=0.7)
    ax_z[0].plot(tm_z, res_trend["v_C"][mask_zoom], label="3. Nowy Trend MF-BB", color="tab:red", lw=1.0, alpha=0.95)
    ax_z[0].set_ylabel("u_dc [V]")
    ax_z[0].grid(True, alpha=0.3)
    ax_z[0].legend(loc="lower right")
    ax_z[0].set_title("Powiekszenie (Zoom 140–210 ms) przy probkowaniu 200 kHz: skok 240->160V i wylaczenie obciazenia")

    ax_z[1].plot(tm_z, res_mbb["i_L"][mask_zoom], label="2. Model-Based BB", color="tab:blue", lw=0.8, alpha=0.85)
    ax_z[1].plot(tm_z, res_mfbb["i_L"][mask_zoom], label="1. Klasyczny MF-BB", color="tab:green", lw=0.8, ls="--", alpha=0.7)
    ax_z[1].plot(tm_z, res_trend["i_L"][mask_zoom], label="3. Nowy Trend MF-BB", color="tab:red", lw=0.8, alpha=0.95)
    ax_z[1].set_ylabel("i_L [A]")
    ax_z[1].set_xlabel("czas [ms]")
    ax_z[1].grid(True, alpha=0.3)
    ax_z[1].legend(loc="upper right")

    for t_ev in [150, 200]:
        ax_z[0].axvline(t_ev, color="k", lw=0.7, ls=":", alpha=0.5)
        ax_z[1].axvline(t_ev, color="k", lw=0.7, ls=":", alpha=0.5)

    fig_z.tight_layout()
    out_png_zoom = "optymalizacja/porownanie_3metody_200khz_zoom.png"
    fig_z.savefig(out_png_zoom, dpi=130)
    print(f"Zapisano wykres zoom: {out_png_zoom}")


if __name__ == "__main__":
    main()
