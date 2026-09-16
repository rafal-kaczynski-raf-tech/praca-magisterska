"""Optymalizacja PSO regulatora Trend MF-BB z uzyciem NOWEJ dwustrefowej funkcji celu
prof. Iwanskiego (error_threshold_aware).

Koncepcja:
- Jesli |e_u| > 5V (stan przejsciowy / duzy blad):
    cel to nasycenie pradu i_L -> +i_max lub -i_max.
    Koszt nie sprawdza bledu napiecia, tylko precyzje osiagniecia pradu nasycenia:
    J_trans = integral( |i_target_sat - i_L| ) dt
- Jesli |e_u| <= 5V (stan ustalony / maly blad):
    cel to dokladnosc napiecia i minimalizacja tetnien pradu:
    J_steady = IAE_u + lam * IAE_i
- Koszt calkowity:
    J = w_trans * J_trans + J_steady   (w_trans=40.0 wyrownuje rzedy wielkosci 40:1)

Parametry obwodu: L = 750 uH, C = 200 uF, R_load = 47.619 Ohm, Ts = 10 us (100 kHz).
Horyzont: 0.30 s (scenariusz prof. Iwanskiego).

Uruchomienie:  python -m optymalizacja.pso_trend_threshold
"""
from __future__ import annotations
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ConverterParams, TrendMFBBParams, SimulationConfig
from src.simulator import Simulator
from src.filters import tustin_1st_order_coeffs
from optymalizacja.scenarios import professor_scenario, PROF_T_END
from optymalizacja.cost_functions import error_threshold_aware
from optymalizacja.pso import _build_u_ref_arr
from optymalizacja.analiza_scenariusz_prof import analyse as analyse_prof

N_PARTICLES = 20
N_ITER = 35
W_MAX = 0.9
W_MIN = 0.4
C1 = 1.49445
C2 = 1.49445
SEED = 42

# Granice przestrzeni poszukiwan: wi in [0.05, 5.0], wv in [0.0005, 10.0], fci in [50, 10000], fcu in [50, 10000]
LB = np.array([np.log10(0.05), np.log10(0.0005), np.log10(50.0), np.log10(50.0)])
UB = np.array([np.log10(5.0),  np.log10(10.0),   np.log10(10000.0), np.log10(10000.0)])


def evaluate_threshold(x: np.ndarray, u_thresh: float = 5.0, w_trans: float = 40.0) -> float:
    wi = 10.0 ** x[0]
    wv = 10.0 ** x[1]
    fc_i = 10.0 ** x[2]
    fc_u = 10.0 ** x[3]

    conv = ConverterParams(
        V_in=100.0,
        L=0.75e-3,
        C=200e-6,           # 200 uF zadanego przez profesora
        R_load=47.619,
        v_C0=100.0,
        i_L0=0.0,
    )
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,     # Ts = 10 us (100 kHz) zgodnie z potwierdzeniem
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
    res = sim.run()
    t = res["t"]
    u_ref_arr = _build_u_ref_arr(t, 240.0, professor_scenario())
    return error_threshold_aware(
        t, res["v_C"], res["i_L"], res["s"], u_ref_arr,
        i_des=res["i_des_phys"], u_thresh=u_thresh, w_trans=w_trans, lam=1.0
    )


def run_pso(u_thresh: float = 5.0, w_trans: float = 40.0, verbose: bool = True) -> dict:
    rng = np.random.default_rng(SEED)
    dim = 4
    v_max = (UB - LB) * 0.2

    X = rng.uniform(LB, UB, size=(N_PARTICLES, dim))
    # Punkt bazowy z poprzedniego optimum jako jedna z czastek
    X[0] = np.array([np.log10(0.3384), np.log10(0.7194), np.log10(922.9), np.log10(235.1)])
    V = rng.uniform(-v_max, v_max, size=(N_PARTICLES, dim))

    if verbose:
        print(f"Ewaluacja roju ({N_PARTICLES} czastek, u_thresh={u_thresh}V, w_trans={w_trans})...")

    F = np.array([evaluate_threshold(x, u_thresh=u_thresh, w_trans=w_trans) for x in X])
    pbest_X = X.copy()
    pbest_F = F.copy()

    g_idx = int(np.argmin(pbest_F))
    gbest_F = float(pbest_F[g_idx])
    gbest_X = pbest_X[g_idx].copy()

    history_gbest_F = np.zeros(N_ITER + 1)
    history_gbest_F[0] = gbest_F

    if verbose:
        print(f"  iter  0  J* = {gbest_F:.5f}  (wi={10**gbest_X[0]:.4f}, wv={10**gbest_X[1]:.4f}, fci={10**gbest_X[2]:.0f}Hz, fcu={10**gbest_X[3]:.0f}Hz)")

    t0 = time.time()
    for it in range(1, N_ITER + 1):
        w = W_MAX - (W_MAX - W_MIN) * (it - 1) / max(1, N_ITER - 1)

        r1 = rng.random((N_PARTICLES, dim))
        r2 = rng.random((N_PARTICLES, dim))
        V = w * V + C1 * r1 * (pbest_X - X) + C2 * r2 * (gbest_X[None, :] - X)
        V = np.clip(V, -v_max, v_max)

        X = X + V
        for d in range(dim):
            below = X[:, d] < LB[d]
            above = X[:, d] > UB[d]
            X[below, d] = LB[d] + (LB[d] - X[below, d])
            X[above, d] = UB[d] - (X[above, d] - UB[d])
            V[below, d] *= -0.5
            V[above, d] *= -0.5
        X = np.clip(X, LB, UB)

        F = np.array([evaluate_threshold(x, u_thresh=u_thresh, w_trans=w_trans) for x in X])

        improved = F < pbest_F
        pbest_X[improved] = X[improved]
        pbest_F[improved] = F[improved]

        g_idx = int(np.argmin(pbest_F))
        if pbest_F[g_idx] < gbest_F:
            gbest_F = float(pbest_F[g_idx])
            gbest_X = pbest_X[g_idx].copy()

        history_gbest_F[it] = gbest_F

        if verbose and (it % 5 == 0 or it == N_ITER):
            print(f"  iter {it:2d}  J* = {gbest_F:.5f}  (wi={10**gbest_X[0]:.4f}, wv={10**gbest_X[1]:.4f}, fci={10**gbest_X[2]:.0f}Hz, fcu={10**gbest_X[3]:.0f}Hz)")

    print(f"PSO zakonczone w {time.time() - t0:.1f} s.")

    return {
        "wi_opt": float(10.0 ** gbest_X[0]),
        "wv_opt": float(10.0 ** gbest_X[1]),
        "fc_i_opt": float(10.0 ** gbest_X[2]),
        "fc_u_opt": float(10.0 ** gbest_X[3]),
        "gbest_F": gbest_F,
        "history_gbest_F": history_gbest_F,
    }


def simulate(wi: float, wv: float, fci: float, fcu: float) -> dict:
    conv = ConverterParams(V_in=100.0, L=0.75e-3, C=200e-6, R_load=47.619, v_C0=100.0, i_L0=0.0)
    ctrl = TrendMFBBParams(u_ref=240.0, T_s_ctrl=10e-6, wi=wi, wv=wv, fc_i=fci, fc_u=fcu, i_max=20.0, i_min=-20.0)
    cfg = SimulationConfig(converter=conv, controller=ctrl, dt_phys=0.5e-6, T_end=PROF_T_END, scenario=professor_scenario())
    return Simulator(cfg).run()


def main():
    print("=" * 80)
    print("OPTYMALIZACJA PSO DLA DWUSTREFOWEJ FUNKCJI CELU PROF. IWANSKIEGO")
    print("Obwod: L=750uH, C=200uF, Ts=10us (100 kHz), T_end=0.30s")
    print("=" * 80)

    res_pso = run_pso(u_thresh=5.0, w_trans=40.0, verbose=True)

    wi_new = res_pso["wi_opt"]
    wv_new = res_pso["wv_opt"]
    fci_new = res_pso["fc_i_opt"]
    fcu_new = res_pso["fc_u_opt"]

    print("\n" + "=" * 80)
    print("NOWE NASTAWY WYBRANE PRZEZ PSO DLA DWUSTREFOWEJ FUNKCJI CELU:")
    print(f"  w_i*  = {wi_new:.4f}")
    print(f"  w_v*  = {wv_new:.4f}")
    print(f"  fc_i* = {fci_new:.1f} Hz")
    print(f"  fc_u* = {fcu_new:.1f} Hz")
    print("=" * 80)

    # 1. Symulacja z nowymi nastawami
    res_new = simulate(wi_new, wv_new, fci_new, fcu_new)

    # 2. Symulacja ze starymi nastawami (zwyciezca ze starej funkcji CurrentAware: wi=0.3384, wv=0.7194)
    res_old = simulate(0.3384, 0.7194, 922.9, 235.1)

    # Porownanie zapadu napiecia w t=150ms i przeregulowania w t=100ms
    t = res_new["t"]
    v_new = res_new["v_C"]
    v_old = res_old["v_C"]
    i_new = res_new["i_L"]
    i_old = res_old["i_L"]

    m150 = (t >= 0.150) & (t <= 0.155)
    undershoot_old = 160.0 - np.min(v_old[m150])
    undershoot_new = 160.0 - np.min(v_new[m150])

    m100 = (t >= 0.100) & (t <= 0.108)
    overshoot_old = np.max(v_old[m100]) - 240.0
    overshoot_new = np.max(v_new[m100]) - 240.0

    print("\nPOROWNANIE UDERZEN W NAPIECIU (Stara funkcja vs Nowa funkcja prof. Iwanskiego):")
    print(f"  Skok 240->160V: Zapad napiecia (undershoot): Stara = {undershoot_old:.2f} V | Nowa = {undershoot_new:.2f} V")
    print(f"  Wlaczenie R   : Przereg. (overshoot):        Stara = {overshoot_old:.2f} V | Nowa = {overshoot_new:.2f} V")

    # Analiza stanow ustalonych
    rows_new = analyse_prof(res_new, 240.0, 0.15, 160.0)
    rows_old = analyse_prof(res_old, 240.0, 0.15, 160.0)

    print("\nTETNIENIA PRADU W STANIE USTALONYM (240V ON, segment [0.10, 0.15) s):")
    print(f"  Stara funkcja celu: iL_pp = {rows_old[1]['iL_ripple_pp_A']:.3f} A ({rows_old[1]['iL_ripple_pct']:.2f}%), v_err = {rows_old[1]['v_err_V']:+.3f} V")
    print(f"  Nowa funkcja celu : iL_pp = {rows_new[1]['iL_ripple_pp_A']:.3f} A ({rows_new[1]['iL_ripple_pct']:.2f}%), v_err = {rows_new[1]['v_err_V']:+.3f} V")

    # Rysowanie wykresu porownawczego
    tm = t * 1e3
    m_z = (t >= 0.145) & (t <= 0.160)
    tm_z = t[m_z] * 1e3

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(tm_z, v_old[m_z], label=f"Stare nastawy (CurrentAware: wv=0.72)", color="tab:blue", lw=0.9, ls="--")
    ax[0].plot(tm_z, v_new[m_z], label=f"Nowe nastawy (Dwustrefowa: wv={wv_new:.2f})", color="tab:red", lw=1.0)
    ax[0].axhline(160.0, color="k", lw=0.7, ls=":", label="u_ref = 160V")
    ax[0].set_ylabel("u_dc [V]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="upper right")
    ax[0].set_title("Wplyw dwustrefowej funkcji celu prof. Iwanskiego na skok 240V -> 160V (@150ms)")

    ax[1].plot(tm_z, i_old[m_z], label="Stare nastawy (CurrentAware)", color="tab:blue", lw=0.8, ls="--")
    ax[1].plot(tm_z, i_new[m_z], label="Nowe nastawy (Dwustrefowa)", color="tab:red", lw=0.9)
    ax[1].axhline(-20.0, color="k", lw=0.6, ls=":", label="Limit -20A")
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_xlabel("czas [ms]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="lower right")

    fig.tight_layout()
    out_png = "optymalizacja/wykres_porownanie_funkcji_profesora.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nZapisano wykres: {out_png}")


if __name__ == "__main__":
    main()
