"""Walidacja 1:1 - warstwa OOP vs PSIM (pelne 100 ms).

Identyczny algorytm jak porownanie_bb_estim_full_psim.py, ale uzywa
warstwy OOP (src.simulator.Simulator) zamiast proceduralnego modulu.

Sluzy jako dowod, ze refactoring do OOP nie zmienil zadnej liczby vs PSIM.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import default_config
from src.simulator import Simulator

CSV_PATH = "psim_bb_estim_full.csv"
T_START = 0.080  # okno steady-state [s]
T_END = 0.100


def metrics(name: str, t, v_C, i_L, s, t_start: float, t_end: float) -> dict:
    m = (t >= t_start) & (t <= t_end)
    vC = np.asarray(v_C)[m]
    iL = np.asarray(i_L)[m]
    sm = np.asarray(s)[m]
    return dict(
        name=name,
        vC_mean=vC.mean(), vC_pp=vC.max() - vC.min(), vC_std=vC.std(),
        iL_mean=iL.mean(), iL_std=iL.std(),
        iL_min=iL.min(), iL_max=iL.max(),
        duty=sm.mean(),
    )


def print_compare(m_psim: dict, m_py: dict) -> None:
    print()
    print(f"{'Metryka':<22}{'PSIM':>14}{'Python OOP':>14}{'Δ':>14}{'Δ %':>10}")
    print("-" * 74)
    for k, lbl in [
        ("vC_mean", "udc mean [V]"),
        ("vC_pp", "udc ripple PP [V]"),
        ("vC_std", "udc std [V]"),
        ("iL_mean", "iL mean [A]"),
        ("iL_std", "iL std [A]"),
        ("iL_min", "iL min [A]"),
        ("iL_max", "iL max [A]"),
        ("duty", "duty cycle"),
    ]:
        p, q = m_psim[k], m_py[k]
        d = q - p
        pct = 100.0 * d / p if abs(p) > 1e-9 else float("nan")
        print(f"{lbl:<22}{p:>14.5f}{q:>14.5f}{d:>+14.5f}{pct:>+9.2f}%")


def downsample_to_psim(res: dict, t_psim) -> tuple:
    idx = np.searchsorted(res["t"], t_psim)
    idx = np.clip(idx, 0, len(res["t"]) - 1)
    return res["i_L"][idx], res["v_C"][idx], res["s"][idx]


def main() -> None:
    df = pd.read_csv(CSV_PATH).sort_values("Time").reset_index(drop=True)
    t_psim = df["Time"].to_numpy()
    print(f"PSIM CSV: {len(t_psim)} probek, t = {t_psim[0]*1000:.2f}-{t_psim[-1]*1000:.2f} ms")

    cfg = default_config()
    print(f"R_load OOP: {cfg.converter.R_load:.4f} Ohm, u_ref: {cfg.controller.u_ref} V")
    print(f"\nSymulacja OOP ({cfg.T_end*1000:.0f} ms, {cfg.N_steps} krokow)...")

    sim = Simulator(cfg)
    res = sim.run()

    # UWAGA: metryki liczone na pelnej tablicy Pythona (siatka 0.5us),
    # PSIM na swojej siatce (10us). Tak samo jak porownanie_bb_estim_full_psim.py.
    m_psim = metrics("PSIM", t_psim, df["udc_BB"], df["I(L6)"], df["pwmBB"], T_START, T_END)
    m_py = metrics("OOP", res["t"], res["v_C"], res["i_L"], res["s"], T_START, T_END)

    print(f"\n=== Steady state ({T_START*1000:.0f}-{T_END*1000:.0f} ms) ===")
    print_compare(m_psim, m_py)

    # Dodatkowo: porownanie iout_filt i i_des
    tc = res["t_ctrl"]
    mc = (tc >= T_START) & (tc <= T_END)
    print(f"\niout_filt mean:  PSIM={df['iout_est_filt_BB'][(t_psim>=T_START)&(t_psim<=T_END)].mean():.5f}A   "
          f"Python OOP={res['iout_filt'][mc].mean():.5f}A   "
          f"Δ={res['iout_filt'][mc].mean() - df['iout_est_filt_BB'][(t_psim>=T_START)&(t_psim<=T_END)].mean():+.5f}A")
    print(f"i_des mean:      PSIM={df['i_ref_BB'][(t_psim>=T_START)&(t_psim<=T_END)].mean():.5f}A   "
          f"Python OOP={res['i_des'][mc].mean():.5f}A   "
          f"Δ={res['i_des'][mc].mean() - df['i_ref_BB'][(t_psim>=T_START)&(t_psim<=T_END)].mean():+.5f}A")


if __name__ == "__main__":
    main()
