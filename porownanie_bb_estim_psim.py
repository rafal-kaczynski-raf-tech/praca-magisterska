"""
Walidacja 1:1 — Python (BB with estimation) vs PSIM.

Wczytuje psim_bb_estim_wyniki.csv (eksport z PSIM Simview, wycinek
~208–290 ms steady state), uruchamia symulację Pythona z identycznymi
parametrami i nakłada przebiegi na 1 wykres + liczy metryki zgodności.

Mapowanie kolumn CSV → Python (BB = "with estimation", BB1 = "without"):
  Time              -> t
  udc_BB            -> v_C
  iL_BB             -> i_L (z ZOH 1-sample delay sterownika)
  iout_est_BB       -> iout_est (przed filtrem)
  iout_est_filt_BB  -> iout_filt (po LPF 2kHz)
  i_ref_BB          -> i_des = iout_filt * u_ref / V_in
  pwmBB             -> s
  i_out_M_BB        -> v_C / R_load (rzeczywisty prąd obciążenia z pomiaru)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import demo_bangbang_estim as sim


CSV_PATH = "psim_bb_estim_wyniki.csv"


def load_psim():
    df = pd.read_csv(CSV_PATH).sort_values("Time").reset_index(drop=True)
    return df


def metrics(name, t, v_C, i_L, s, t_start, t_end):
    m = (t >= t_start) & (t <= t_end)
    vC = np.asarray(v_C)[m]
    iL = np.asarray(i_L)[m]
    sm = np.asarray(s)[m]
    return {
        "name":   name,
        "vC_mean": vC.mean(),
        "vC_pp":   vC.max() - vC.min(),
        "vC_std":  vC.std(),
        "iL_mean": iL.mean(),
        "iL_std":  iL.std(),
        "iL_min":  iL.min(),
        "iL_max":  iL.max(),
        "duty":    sm.mean(),
    }


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


def plot(df_psim, res, t_start, t_end, fname="porownanie_bb_estim_python_vs_psim.png"):
    t_psim = df_psim["Time"].values
    m_p = (t_psim >= t_start) & (t_psim <= t_end)
    t_py = res["t"]
    m_y  = (t_py >= t_start) & (t_py <= t_end)
    t_pyc = res["t_ctrl"]
    m_yc  = (t_pyc >= t_start) & (t_pyc <= t_end)

    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # 1. Napięcie
    ax[0].plot(t_psim[m_p]*1e3, df_psim["udc_BB"].values[m_p],
               label="PSIM udc_BB", color="C0", lw=0.6)
    ax[0].plot(t_py[m_y]*1e3, res["v_C"][m_y],
               label="Python v_C", color="C3", lw=0.6, alpha=0.8)
    ax[0].axhline(sim.u_ref, color="k", ls="--", lw=0.6)
    ax[0].set_ylabel("Napięcie u_dc [V]")
    ax[0].set_title(f"BB with estimation — Python vs PSIM "
                    f"(R_load={sim.R_load:.0f}Ω, u_ref={sim.u_ref:.0f}V, "
                    f"steady-state {t_start*1e3:.0f}–{t_end*1e3:.0f} ms)")
    ax[0].grid(alpha=0.3); ax[0].legend(loc="upper right")

    # 2. Prąd cewki
    ax[1].plot(t_psim[m_p]*1e3, df_psim["iL_BB"].values[m_p],
               label="PSIM iL_BB", color="C0", lw=0.4)
    ax[1].plot(t_py[m_y]*1e3, res["i_L"][m_y],
               label="Python i_L", color="C3", lw=0.4, alpha=0.7)
    ax[1].set_ylabel("Prąd cewki i_L [A]")
    ax[1].grid(alpha=0.3); ax[1].legend(loc="upper right")

    # 3. Estymator prądu obciążenia (przed i po filtrze)
    ax[2].plot(t_psim[m_p]*1e3, df_psim["iout_est_BB"].values[m_p],
               label="PSIM iout_est", color="C0", lw=0.5, alpha=0.5)
    ax[2].plot(t_psim[m_p]*1e3, df_psim["iout_est_filt_BB"].values[m_p],
               label="PSIM iout_filt (LPF)", color="C0", lw=1.2)
    ax[2].plot(t_pyc[m_yc]*1e3, res["iout_est"][m_yc],
               label="Python iout_est", color="C3", lw=0.5, alpha=0.5)
    ax[2].plot(t_pyc[m_yc]*1e3, res["iout_filt"][m_yc],
               label="Python iout_filt", color="C3", lw=1.2)
    ax[2].axhline(sim.u_ref/sim.R_load, color="green", ls=":",
                  lw=0.8, label=f"i_load_true={sim.u_ref/sim.R_load:.4f}A")
    ax[2].set_ylabel("Estymata I_out [A]")
    ax[2].grid(alpha=0.3); ax[2].legend(loc="upper right")

    # 4. PWM
    ax[3].step(t_psim[m_p]*1e3, df_psim["pwmBB"].values[m_p],
               where="post", color="C0", lw=0.5, label="PSIM pwmBB")
    ax[3].step(t_py[m_y]*1e3, res["s"][m_y],
               where="post", color="C3", lw=0.5, alpha=0.6, label="Python s")
    ax[3].set_ylabel("PWM s")
    ax[3].set_xlabel("Czas [ms]")
    ax[3].set_ylim(-0.1, 1.1)
    ax[3].grid(alpha=0.3); ax[3].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    print(f"\nZapisano wykres: {fname}")


if __name__ == "__main__":
    df = load_psim()
    t_start = df["Time"].min()
    t_end   = df["Time"].max()
    print(f"PSIM CSV:  {len(df)} próbek, t = {t_start*1e3:.2f}–{t_end*1e3:.2f} ms")

    print(f"\nUruchamiam symulację Python ({sim.T_END*1e3:.0f} ms, "
          f"{sim.N_steps} kroków po {sim.dt_phys*1e6} µs)...")
    res = sim.simulate()

    # Metryki w identycznym oknie czasowym co dane PSIM
    m_psim = metrics("PSIM",  df["Time"].values, df["udc_BB"].values,
                     df["iL_BB"].values, df["pwmBB"].values, t_start, t_end)
    m_py   = metrics("Python", res["t"], res["v_C"], res["i_L"], res["s"],
                     t_start, t_end)

    print_compare(m_psim, m_py)

    # Estymator
    print()
    mc_p = (df["Time"].values >= t_start) & (df["Time"].values <= t_end)
    mc_y = (res["t_ctrl"] >= t_start) & (res["t_ctrl"] <= t_end)
    print(f"iout_filt mean:  PSIM={df['iout_est_filt_BB'].values[mc_p].mean():.5f}A   "
          f"Python={res['iout_filt'][mc_y].mean():.5f}A   "
          f"Δ={res['iout_filt'][mc_y].mean() - df['iout_est_filt_BB'].values[mc_p].mean():+.5f}A")
    print(f"i_des mean:      PSIM={df['i_ref_BB'].values[mc_p].mean():.5f}A   "
          f"Python={res['i_des'][mc_y].mean():.5f}A   "
          f"Δ={res['i_des'][mc_y].mean() - df['i_ref_BB'].values[mc_p].mean():+.5f}A")

    plot(df, res, t_start, t_end)
