"""
Walidacja 1:1 — Python (BB with estimation) vs PSIM, pełny eksport 100 ms.

Wczytuje psim_bb_estim_full.csv (10000 próbek co 10 µs, u_ref=240V),
uruchamia symulację Pythona z identycznymi parametrami i:
  1. Nakłada przebiegi na 1 wykres (transient 0-10ms + steady state),
  2. Liczy metryki zgodności w steady state (80-100 ms).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import demo_bangbang_estim as sim

CSV_PATH = "psim_bb_estim_full.csv"


def load_psim():
    df = pd.read_csv(CSV_PATH).sort_values("Time").reset_index(drop=True)
    return df


def metrics(name, t, v_C, i_L, s, t_start, t_end):
    m = (t >= t_start) & (t <= t_end)
    vC = np.asarray(v_C)[m]
    iL = np.asarray(i_L)[m]
    sm = np.asarray(s)[m]
    return dict(name=name,
                vC_mean=vC.mean(), vC_pp=vC.max()-vC.min(), vC_std=vC.std(),
                iL_mean=iL.mean(), iL_std=iL.std(),
                iL_min=iL.min(),   iL_max=iL.max(),
                duty=sm.mean())


def print_compare(m_psim, m_py):
    print()
    print(f"{'Metryka':<22}{'PSIM':>14}{'Python':>14}{'Δ':>14}{'Δ %':>10}")
    print("-" * 74)
    keys = [("vC_mean", "udc mean [V]"),
            ("vC_pp",   "udc ripple PP [V]"),
            ("vC_std",  "udc std [V]"),
            ("iL_mean", "iL mean [A]"),
            ("iL_std",  "iL std [A]"),
            ("iL_min",  "iL min [A]"),
            ("iL_max",  "iL max [A]"),
            ("duty",    "duty cycle")]
    for k, lbl in keys:
        p, q = m_psim[k], m_py[k]
        d = q - p
        pct = 100.0 * d / p if abs(p) > 1e-9 else float("nan")
        print(f"{lbl:<22}{p:>14.5f}{q:>14.5f}{d:>+14.5f}{pct:>+9.2f}%")


def downsample_to_psim(res, t_psim):
    """Wyciąga z symulacji Pythona próbki w tych samych chwilach co PSIM."""
    idx = np.searchsorted(res["t"], t_psim)
    idx = np.clip(idx, 0, len(res["t"]) - 1)
    return res["i_L"][idx], res["v_C"][idx], res["s"][idx]


def plot(df, res, fname="porownanie_bb_estim_full_psim.png"):
    t_psim = df["Time"].values

    fig, ax = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

    # 1. Napięcie (cała symulacja)
    ax[0].plot(t_psim*1e3, df["udc_BB"].values, label="PSIM udc_BB", color="C0", lw=0.6)
    ax[0].plot(res["t"]*1e3, res["v_C"],        label="Python v_C", color="C3",
               lw=0.6, alpha=0.7)
    ax[0].axhline(sim.u_ref, color="k", ls="--", lw=0.6)
    ax[0].set_ylabel("u_dc [V]")
    ax[0].set_title(f"BB with estimation - PEŁNA walidacja 100 ms "
                    f"(R={sim.R_load:.3f}Ω, u_ref={sim.u_ref:.0f}V)")
    ax[0].grid(alpha=0.3); ax[0].legend(loc="lower right")

    # 2. Prąd cewki  (UWAGA: I(L6) - cewka modułu BB. I(L7) to inny moduł.)
    ax[1].plot(t_psim*1e3, df["I(L6)"].values, label="PSIM I(L6) [cewka BB]", color="C0", lw=0.4)
    ax[1].plot(res["t"]*1e3, res["i_L"],       label="Python i_L", color="C3",
               lw=0.4, alpha=0.7)
    ax[1].set_ylabel("i_L [A]")
    ax[1].grid(alpha=0.3); ax[1].legend(loc="upper right")

    # 3. Estymator obciążenia (przed i po filtrze)
    ax[2].plot(t_psim*1e3, df["iout_est_BB"].values, label="PSIM iout_est",
               color="C0", lw=0.4, alpha=0.4)
    ax[2].plot(t_psim*1e3, df["iout_est_filt_BB"].values, label="PSIM iout_filt",
               color="C0", lw=1.0)
    ax[2].plot(res["t_ctrl"]*1e3, res["iout_filt"], label="Python iout_filt",
               color="C3", lw=1.0, alpha=0.7)
    ax[2].axhline(sim.u_ref/sim.R_load, color="green", ls=":",
                  label=f"i_load_true={sim.u_ref/sim.R_load:.4f}A")
    ax[2].set_ylabel("I_out estimator [A]")
    ax[2].grid(alpha=0.3); ax[2].legend(loc="upper right")

    # 4. i_des (i_ref)
    ax[3].plot(t_psim*1e3, df["i_ref_BB"].values, label="PSIM i_ref_BB",
               color="C0", lw=0.8)
    ax[3].plot(res["t_ctrl"]*1e3, res["i_des"], label="Python i_des",
               color="C3", lw=0.8, alpha=0.7)
    ax[3].set_ylabel("i_des [A]")
    ax[3].grid(alpha=0.3); ax[3].legend(loc="upper right")

    # 5. PWM
    ax[4].step(t_psim*1e3, df["pwmBB"].values, where="post", color="C0", lw=0.4,
               label="PSIM pwmBB")
    ax[4].step(res["t"]*1e3, res["s"], where="post", color="C3", lw=0.4, alpha=0.6,
               label="Python s")
    ax[4].set_ylabel("PWM s")
    ax[4].set_xlabel("Czas [ms]")
    ax[4].set_ylim(-0.1, 1.1)
    ax[4].grid(alpha=0.3); ax[4].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    print(f"\nZapisano wykres: {fname}")


def plot_zoom(df, res, t_start, t_end, fname="porownanie_bb_estim_zoom.png"):
    """Zoom na fragment dla widoczności tętnień."""
    mp = (df["Time"].values >= t_start) & (df["Time"].values <= t_end)
    my = (res["t"] >= t_start) & (res["t"] <= t_end)
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    ax[0].plot(df["Time"].values[mp]*1e3, df["udc_BB"].values[mp],
               "o-", color="C0", ms=2, lw=0.8, label="PSIM udc_BB")
    ax[0].plot(res["t"][my]*1e3, res["v_C"][my],
               "-", color="C3", lw=0.8, alpha=0.7, label="Python v_C")
    ax[0].set_ylabel("u_dc [V]"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title(f"Zoom {t_start*1e3:.0f}-{t_end*1e3:.0f} ms")

    ax[1].plot(df["Time"].values[mp]*1e3, df["I(L6)"].values[mp],
               "o-", color="C0", ms=2, lw=0.8, label="PSIM I(L6)")
    ax[1].plot(res["t"][my]*1e3, res["i_L"][my],
               "-", color="C3", lw=0.8, alpha=0.7, label="Python i_L")
    ax[1].set_ylabel("i_L [A]"); ax[1].set_xlabel("Czas [ms]")
    ax[1].grid(alpha=0.3); ax[1].legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    print(f"Zapisano zoom: {fname}")


if __name__ == "__main__":
    df = load_psim()
    print(f"PSIM CSV: {len(df)} próbek, t = {df.Time.min()*1e3:.2f}-{df.Time.max()*1e3:.2f} ms")
    print(f"R_load Python: {sim.R_load:.4f} Ω,  u_ref: {sim.u_ref} V")

    print(f"\nSymulacja Python ({sim.T_END*1e3:.0f} ms, {sim.N_steps} kroków po "
          f"{sim.dt_phys*1e6} µs)...")
    res = sim.simulate()

    # Steady state (80-100 ms)
    t_start, t_end = 0.080, 0.100
    m_psim = metrics("PSIM",   df.Time.values, df.udc_BB.values,
                     df["I(L6)"].values, df.pwmBB.values, t_start, t_end)
    m_py   = metrics("Python", res["t"], res["v_C"], res["i_L"], res["s"],
                     t_start, t_end)
    print(f"\n=== Steady state ({t_start*1e3:.0f}-{t_end*1e3:.0f} ms) ===")
    print_compare(m_psim, m_py)

    # Estymator
    mc_p = (df.Time.values >= t_start) & (df.Time.values <= t_end)
    mc_y = (res["t_ctrl"] >= t_start) & (res["t_ctrl"] <= t_end)
    print(f"\niout_filt mean:  PSIM={df.iout_est_filt_BB.values[mc_p].mean():.5f}A   "
          f"Python={res['iout_filt'][mc_y].mean():.5f}A   "
          f"Δ={res['iout_filt'][mc_y].mean() - df.iout_est_filt_BB.values[mc_p].mean():+.5f}A")
    print(f"i_ref mean:      PSIM={df.i_ref_BB.values[mc_p].mean():.5f}A   "
          f"Python={res['i_des'][mc_y].mean():.5f}A   "
          f"Δ={res['i_des'][mc_y].mean() - df.i_ref_BB.values[mc_p].mean():+.5f}A")

    plot(df, res)
    plot_zoom(df, res, 0.000, 0.005)
    plot_zoom(df, res, 0.090, 0.095, fname="porownanie_bb_estim_zoom_steady.png")
