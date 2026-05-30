"""Wizualizacja zbieznosci PSO + porownanie z grid search.

Generuje:
  - pso_zbieznosc.png: krzywa zbieznosci gbest_F vs iteracja
  - pso_roj_na_mapie.png: ruch roju na mapie konturowej J(wi, fc) z gridu
  - pso_vs_grid.png: porownanie przebiegow optimum PSO vs grid

Uruchomienie:
    python -m optymalizacja.pso_plots
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.scenarios import hard_scenario, HARD_T_END

PSO_PATH = "optymalizacja/pso_results.npz"
GRID_PATH = "optymalizacja/grid_results.npz"
SCENARIO = hard_scenario()
T_END = HARD_T_END


def plot_convergence(out: str = "optymalizacja/pso_zbieznosc.png") -> None:
    pso = np.load(PSO_PATH)
    F = pso["history_gbest_F"]
    iters = np.arange(len(F))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, F, "o-", linewidth=1.5, markersize=4, color="C0")
    ax.set_xlabel("iteracja")
    ax.set_ylabel(f"$J_{{best}}$  ({pso['cost_name']})")
    ax.set_yscale("log")
    ax.set_title(f"Zbieznosc PSO -- funkcja celu {pso['cost_name']}, N=20 czastek")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(float(F[-1]), color="r", linestyle="--", linewidth=0.8,
               label=f"$J^*$ = {float(F[-1]):.4g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out}")


def plot_swarm_on_map(out: str = "optymalizacja/pso_roj_na_mapie.png") -> None:
    """Mapa konturowa J(wi, fc) z trajektoria roju nalozona."""
    pso = np.load(PSO_PATH)
    grid = np.load(GRID_PATH)
    cost_name = str(pso["cost_name"])

    wi_range = grid["wi_range"]
    fc_range = grid["fc_range"]
    J = grid[f"cost_{cost_name}"]
    WI, FC = np.meshgrid(wi_range, fc_range, indexing="ij")

    fig, ax = plt.subplots(figsize=(10, 7))
    cs = ax.contourf(WI, FC, np.log10(np.where(J > 0, J, np.nan)),
                     levels=20, cmap="viridis", alpha=0.85)
    plt.colorbar(cs, ax=ax, label=f"$\\log_{{10}} J$  ({cost_name})")

    # Trajektoria gbest
    hist_X = pso["history_gbest_X"]
    wi_gb = 10 ** hist_X[:, 0]
    fc_gb = 10 ** hist_X[:, 1]
    ax.plot(wi_gb, fc_gb, "-", color="red", linewidth=1.5, alpha=0.7,
            label="trajektoria gbest")

    # Wszystkie pozycje czastek (snapshot kazdej iteracji w odcieniach szarosci)
    swarm = pso["history_X"]   # (n_iter+1, N, 2)
    n_it = swarm.shape[0]
    for it in range(0, n_it, max(1, n_it // 10)):
        alpha = 0.15 + 0.5 * (it / max(1, n_it - 1))
        wi_s = 10 ** swarm[it, :, 0]
        fc_s = 10 ** swarm[it, :, 1]
        ax.scatter(wi_s, fc_s, s=18, c="white", edgecolors="black",
                   alpha=alpha, linewidths=0.5)

    # Optimum gridu (zolta gwiazda) + optimum PSO (czerwona gwiazda)
    i_min, j_min = np.unravel_index(np.nanargmin(J), J.shape)
    ax.plot(wi_range[i_min], fc_range[j_min], "*", color="yellow",
            markersize=22, markeredgecolor="black",
            label=f"grid opt: J={J[i_min,j_min]:.4g}")
    ax.plot(float(pso["wi_opt"]), float(pso["fc_opt"]), "*", color="red",
            markersize=22, markeredgecolor="white",
            label=f"PSO opt: J={float(pso['gbest_F']):.4g}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$w_i$ [-]")
    ax.set_ylabel(r"$f_c$ [Hz]")
    ax.set_title(f"Trajektoria roju PSO na mapie $J$ ({cost_name})")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out}")


def plot_pso_vs_grid(out: str = "optymalizacja/pso_vs_grid.png") -> None:
    pso = np.load(PSO_PATH)
    grid = np.load(GRID_PATH)
    cost_name = str(pso["cost_name"])

    # Optimum gridu
    J = grid[f"cost_{cost_name}"]
    i_min, j_min = np.unravel_index(np.nanargmin(J), J.shape)
    wi_g = float(grid["wi_range"][i_min])
    fc_g = float(grid["fc_range"][j_min])

    wi_p = float(pso["wi_opt"])
    fc_p = float(pso["fc_opt"])

    def simulate(wi, fc):
        base = default_config()
        cfg = replace(base, controller=replace(base.controller, wi=wi, fc_lpf=fc),
                      scenario=SCENARIO, T_end=T_END)
        return Simulator(cfg).run()

    res_g = simulate(wi_g, fc_g)
    res_p = simulate(wi_p, fc_p)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(res_g["t"] * 1e3, res_g["v_C"], color="C1", linewidth=1.0,
             label=f"Grid: wi={wi_g:.3f}, fc={fc_g:.0f}Hz")
    ax1.plot(res_p["t"] * 1e3, res_p["v_C"], color="C0", linewidth=1.0,
             label=f"PSO:  wi={wi_p:.3f}, fc={fc_p:.0f}Hz")
    ax1.axhline(240, ls="--", color="k", alpha=0.4, linewidth=0.8)
    if SCENARIO is not None and SCENARIO.ref_step_value is not None:
        ax1.axhline(SCENARIO.ref_step_value, ls="--", color="gray",
                    alpha=0.4, linewidth=0.8)
    if SCENARIO is not None and SCENARIO.load_step_time is not None:
        for ax in (ax1, ax2):
            ax.axvline(SCENARIO.load_step_time * 1e3, ls=":", color="blue", alpha=0.4)
    if SCENARIO is not None and SCENARIO.ref_step_time is not None:
        for ax in (ax1, ax2):
            ax.axvline(SCENARIO.ref_step_time * 1e3, ls=":", color="red", alpha=0.4)
    ax1.set_ylabel("$u_{dc}$ [V]")
    ax1.set_title(f"Porownanie optimum PSO vs grid ({cost_name})")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(res_g["t"] * 1e3, res_g["i_L"], color="C1", linewidth=1.0)
    ax2.plot(res_p["t"] * 1e3, res_p["i_L"], color="C0", linewidth=1.0)
    ax2.axhline(20, ls="--", color="r", alpha=0.4, linewidth=0.8)
    ax2.set_ylabel("$i_L$ [A]")
    ax2.set_xlabel("czas [ms]")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out}")


def main() -> None:
    plot_convergence()
    plot_swarm_on_map()
    plot_pso_vs_grid()


if __name__ == "__main__":
    main()
