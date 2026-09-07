"""Optymalizacja PSO regulatora Model-Free Bang-Bang z filtrowanym trendem napiecia.

Optymalizuje 4 parametry w przestrzeni logarytmicznej:
  x = [log10(wi), log10(wv), log10(fc_i), log10(fc_u)]

Zgodnie z nota techniczna prof. Iwanskiego (2026):
  - ides filtrowane LPF 1. rzedu z pradu iL (Tustin, pasmo fc_i)
  - uf filtrowane LPF 1. rzedu z napiecia uout (Tustin, pasmo fc_u)
  - delta_uf = uf[k] - uf[k-1]
  - eBB = (u_ref - uout) + wi*(ides - iL) - wv*delta_uf

Uruchomienie:  python -m optymalizacja.pso_trend_mfbb
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ConverterParams, TrendMFBBParams, SimulationConfig
from src.simulator import Simulator
from src.filters import tustin_1st_order_coeffs
from optymalizacja.scenarios import professor_scenario, PROF_T_END
from optymalizacja.cost_functions import current_aware
from optymalizacja.pso import _build_u_ref_arr

# Hiperparametry PSO (zgodne z dotychczasowymi badaniami)
N_PARTICLES = 20
N_ITER = 35
W_MAX = 0.9
W_MIN = 0.4
C1 = 1.49445
C2 = 1.49445
SEED = 42

# Granice przestrzeni poszukiwan
# wi:   [0.05, 5.0]
# wv:   [0.0005, 5.0]
# fc_i: [50.0, 10000.0] Hz
# fc_u: [50.0, 10000.0] Hz
LB = np.array([np.log10(0.05), np.log10(0.0005), np.log10(50.0), np.log10(50.0)])
UB = np.array([np.log10(5.0),  np.log10(5.0),    np.log10(10000.0), np.log10(10000.0)])


def evaluate_trend(
    x: np.ndarray,
    C_val: float = 1500e-6,
    lam: float = 1.0,
) -> float:
    """Ewaluacja funkcji celu CurrentAware dla wektora x w skali log10."""
    wi = 10.0 ** x[0]
    wv = 10.0 ** x[1]
    fc_i = 10.0 ** x[2]
    fc_u = 10.0 ** x[3]

    conv = ConverterParams(
        V_in=100.0,
        L=0.75e-3,
        C=C_val,
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
    res = sim.run()
    t = res["t"]
    u_ref_arr = _build_u_ref_arr(t, 240.0, professor_scenario())
    return current_aware(
        t, res["v_C"], res["i_L"], res["s"], u_ref_arr,
        i_des=res["i_des_phys"], lam=lam
    )


def run_pso_trend(
    C_val: float = 1500e-6,
    lam: float = 1.0,
    n_particles: int = N_PARTICLES,
    n_iter: int = N_ITER,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    dim = 4
    v_max = (UB - LB) * 0.2

    # Inicjalizacja pozycji i predkosci
    X = rng.uniform(LB, UB, size=(n_particles, dim))
    # Dodajmy punkt bazowy czata jako jedna z czastek do porownania
    X[0] = np.array([np.log10(0.2), np.log10(0.05), np.log10(321.0), np.log10(321.0)])

    V = rng.uniform(-v_max, v_max, size=(n_particles, dim))

    if verbose:
        print(f"Ewaluacja roju poczatkowego ({n_particles} czastek)...")

    F = np.array([evaluate_trend(x, C_val=C_val, lam=lam) for x in X])
    pbest_X = X.copy()
    pbest_F = F.copy()

    g_idx = int(np.argmin(pbest_F))
    gbest_F = float(pbest_F[g_idx])
    gbest_X = pbest_X[g_idx].copy()

    history_gbest_F = np.zeros(n_iter + 1)
    history_gbest_F[0] = gbest_F

    if verbose:
        wi0 = 10.0 ** gbest_X[0]
        wv0 = 10.0 ** gbest_X[1]
        fci0 = 10.0 ** gbest_X[2]
        fcu0 = 10.0 ** gbest_X[3]
        print(f"  iter  0  J* = {gbest_F:.5f}  (wi={wi0:.4f}, wv={wv0:.4f}, fci={fci0:.0f}Hz, fcu={fcu0:.0f}Hz)")

    t0 = time.time()
    for it in range(1, n_iter + 1):
        w = W_MAX - (W_MAX - W_MIN) * (it - 1) / max(1, n_iter - 1)

        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
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

        F = np.array([evaluate_trend(x, C_val=C_val, lam=lam) for x in X])

        improved = F < pbest_F
        pbest_X[improved] = X[improved]
        pbest_F[improved] = F[improved]

        g_idx = int(np.argmin(pbest_F))
        if pbest_F[g_idx] < gbest_F:
            gbest_F = float(pbest_F[g_idx])
            gbest_X = pbest_X[g_idx].copy()

        history_gbest_F[it] = gbest_F

        if verbose and (it % 5 == 0 or it == n_iter):
            wib = 10.0 ** gbest_X[0]
            wvb = 10.0 ** gbest_X[1]
            fcib = 10.0 ** gbest_X[2]
            fcub = 10.0 ** gbest_X[3]
            print(f"  iter {it:2d}  J* = {gbest_F:.5f}  (wi={wib:.4f}, wv={wvb:.4f}, fci={fcib:.0f}Hz, fcu={fcub:.0f}Hz)")

    elapsed = time.time() - t0
    if verbose:
        print(f"Optymalizacja zakonczona w {elapsed:.1f} s.")

    opt_params = {
        "wi_opt": float(10.0 ** gbest_X[0]),
        "wv_opt": float(10.0 ** gbest_X[1]),
        "fc_i_opt": float(10.0 ** gbest_X[2]),
        "fc_u_opt": float(10.0 ** gbest_X[3]),
        "gbest_F": gbest_F,
        "history_gbest_F": history_gbest_F,
    }
    return opt_params


def simulate_with_params(wi: float, wv: float, fc_i: float, fc_u: float, C_val: float = 1500e-6) -> dict:
    conv = ConverterParams(
        V_in=100.0,
        L=0.75e-3,
        C=C_val,
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
    return Simulator(cfg).run()


def main():
    print("=" * 78)
    print("OPTYMALIZACJA PSO: Trend MF-BB (w_i, w_v, fc_i, fc_u)")
    print("Scenariusz dynamiczny prof. Iwanskiego (0.30s), C = 1500 uF")
    print("=" * 78)

    opt = run_pso_trend(C_val=1500e-6, lam=1.0, verbose=True)

    wi_opt = opt["wi_opt"]
    wv_opt = opt["wv_opt"]
    fc_i_opt = opt["fc_i_opt"]
    fc_u_opt = opt["fc_u_opt"]
    j_opt = opt["gbest_F"]

    # Oblicz dyskretne wspolczynniki filtru (Tustin, Ts=10 us)
    Ts = 10e-6
    fs = 1.0 / Ts
    alpha_i, beta_i = tustin_1st_order_coeffs(fc_i_opt, fs)
    alpha_u, beta_u = tustin_1st_order_coeffs(fc_u_opt, fs)

    print("\n" + "=" * 78)
    print("WYNIKI OPTYMALIZACJI PSO DLA TREND MF-BB:")
    print("=" * 78)
    print(f"  w_i*       = {wi_opt:.4f}")
    print(f"  w_v*       = {wv_opt:.4f}")
    print(f"  fc_i*      = {fc_i_opt:.1f} Hz  --> alpha_i={alpha_i:.5f}, beta_i={beta_i:.5f}")
    print(f"  fc_u*      = {fc_u_opt:.1f} Hz  --> alpha_u={alpha_u:.5f}, beta_u={beta_u:.5f}")
    print(f"  Koszt J*   = {j_opt:.5f}")

    # Porownanie: Baza (wi=0.2, wv=0.0, fc=321Hz) vs Optimum PSO
    print("\nSymulacja wariantu bazowego (bez trendu, fc=321Hz)...")
    res_base = simulate_with_params(wi=0.2, wv=0.0, fc_i=321.0, fc_u=321.0, C_val=1500e-6)

    print("Symulacja wariantu zoptymalizowanego przez PSO...")
    res_opt = simulate_with_params(wi=wi_opt, wv=wv_opt, fc_i=fc_i_opt, fc_u=fc_u_opt, C_val=1500e-6)

    # Zapis wynikow
    out_npz = "optymalizacja/pso_trend_mfbb_results.npz"
    np.savez(
        out_npz,
        wi_opt=wi_opt,
        wv_opt=wv_opt,
        fc_i_opt=fc_i_opt,
        fc_u_opt=fc_u_opt,
        alpha_i=alpha_i,
        beta_i=beta_i,
        alpha_u=alpha_u,
        beta_u=beta_u,
        gbest_F=j_opt,
        history_gbest_F=opt["history_gbest_F"],
    )
    print(f"Zapisano wyniki PSO: {out_npz}")

    # Rysowanie wykresu porownawczego
    t = res_base["t"] * 1e3
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax[0].plot(t, res_base["v_C"], label="MF-BB bazowy (wi=0.2, wv=0, fc=321Hz)", color="tab:blue", lw=0.9, alpha=0.8)
    ax[0].plot(t, res_opt["v_C"], label=f"Trend MF-BB (PSO: wi={wi_opt:.3f}, wv={wv_opt:.3f})", color="tab:orange", lw=1.0)
    ax[0].set_ylabel("u_dc [V]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="lower right")
    ax[0].set_title("Porownanie: Bazowy MF-BB vs Trend MF-BB zoptymalizowany przez PSO")

    ax[1].plot(t, res_base["i_L"], label="MF-BB bazowy", color="tab:blue", lw=0.8, alpha=0.7)
    ax[1].plot(t, res_opt["i_L"], label=f"Trend MF-BB (fc_i={fc_i_opt:.0f}Hz, fc_u={fc_u_opt:.0f}Hz)", color="tab:orange", lw=0.85)
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_xlabel("czas [ms]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right")

    fig.tight_layout()
    out_png = "optymalizacja/wykres_trend_mfbb_pso.png"
    fig.savefig(out_png, dpi=130)
    print(f"Zapisano wykres porownawczy: {out_png}")


if __name__ == "__main__":
    main()
