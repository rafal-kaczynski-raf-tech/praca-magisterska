"""Analiza wynikow grid search: mapy konturowe + tabela optymalnych nastaw.

Uruchomienie:
    python -m optymalizacja.analyze_results
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.cost_functions import COST_FUNCTIONS
from optymalizacja.scenarios import hard_scenario, HARD_T_END


GRID_PATH = "optymalizacja/grid_results.npz"
SCENARIO = hard_scenario()
T_END = HARD_T_END


def load_grid() -> dict:
    data = np.load(GRID_PATH)
    return {key: data[key] for key in data.files}


def plot_cost_maps(grid: dict, out_path: str = "optymalizacja/mapy_kosztow.png") -> None:
    """Mapy konturowe log10(J) dla kazdej funkcji celu."""
    wi_range = grid["wi_range"]
    fc_range = grid["fc_range"]
    cost_names = list(COST_FUNCTIONS.keys())
    n_cost = len(cost_names)

    n_cols = 3
    n_rows = (n_cost + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.atleast_2d(axes).ravel()

    for k, name in enumerate(cost_names):
        ax = axes[k]
        J = grid[f"cost_{name}"]
        # log10 dla wizualnej rozrzutu (J zmienia sie o rzedy wielkosci)
        # ochrona przed zerami
        J_safe = np.where(J > 0, J, np.nan)
        Z = np.log10(J_safe)

        # contourf -- WI na osi X, FC na osi Y -> trzeba transpose
        # bo grid[i, j] = J(wi[i], fc[j])
        WI, FC = np.meshgrid(wi_range, fc_range, indexing="ij")
        cs = ax.contourf(WI, FC, Z, levels=20, cmap="viridis")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$w_i$ [-]")
        ax.set_ylabel(r"$f_c$ [Hz]")
        ax.set_title(f"{name}:  $\\log_{{10}} J$")
        plt.colorbar(cs, ax=ax)

        # zaznacz minimum
        i_min, j_min = np.unravel_index(np.nanargmin(J), J.shape)
        ax.plot(wi_range[i_min], fc_range[j_min], "r*", markersize=15,
                markeredgecolor="white")

    for k in range(n_cost, len(axes)):
        axes[k].axis("off")

    fig.suptitle("Krajobraz funkcji celu dla MF-BB (grid 2D: $w_i \\times f_c$)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_path}")


def optimal_table(grid: dict) -> list:
    """Tabela: dla kazdej funkcji celu zwraca optymalne (wi*, fc*) + metryki."""
    wi_range = grid["wi_range"]
    fc_range = grid["fc_range"]
    rows = []
    for name in COST_FUNCTIONS.keys():
        J = grid[f"cost_{name}"]
        i_min, j_min = np.unravel_index(np.nanargmin(J), J.shape)
        wi_opt = float(wi_range[i_min])
        fc_opt = float(fc_range[j_min])
        rows.append({
            "cost_name": name,
            "wi_opt": wi_opt,
            "fc_opt": fc_opt,
            "J_min": float(J[i_min, j_min]),
            "M_p_pct": float(grid["M_p_pct"][i_min, j_min]),
            "t_s_ms": float(grid["t_s_ms"][i_min, j_min]),
            "e_ss_V": float(grid["e_ss_V"][i_min, j_min]),
            "ripple_V": float(grid["ripple_V"][i_min, j_min]),
            "iL_peak_A": float(grid["iL_peak_A"][i_min, j_min]),
            "n_switch": float(grid["n_switch"][i_min, j_min]),
        })
    return rows


def print_table(rows: list) -> None:
    """Tabela tekstowa do konsoli."""
    headers = ["Cost", "wi*", "fc* [Hz]", "M_p [%]", "t_s [ms]",
               "e_ss [V]", "ripple [V]", "iL_max [A]", "n_switch"]
    fmt = "{:<11} {:>7} {:>9} {:>8} {:>9} {:>8} {:>10} {:>10} {:>9}"
    print("\n" + "=" * 100)
    print("OPTYMALNE NASTAWY DLA KAZDEJ FUNKCJI CELU")
    print("=" * 100)
    print(fmt.format(*headers))
    print("-" * 100)
    for r in rows:
        print(fmt.format(
            r["cost_name"],
            f"{r['wi_opt']:.3g}",
            f"{r['fc_opt']:.0f}",
            f"{r['M_p_pct']:.2f}",
            f"{r['t_s_ms']:.2f}",
            f"{r['e_ss_V']:.3f}",
            f"{r['ripple_V']:.3f}",
            f"{r['iL_peak_A']:.2f}",
            f"{int(r['n_switch'])}",
        ))
    print("=" * 100)


def plot_trajectories(rows: list, out_path: str = "optymalizacja/przebiegi_optimum.png") -> None:
    """Nakladka przebiegow u_dc(t) dla optymalnych nastaw kazdej funkcji celu."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))

    for r, c in zip(rows, colors):
        base = default_config()
        new_ctrl = replace(base.controller, wi=r["wi_opt"], fc_lpf=r["fc_opt"])
        cfg = replace(base, controller=new_ctrl, scenario=SCENARIO, T_end=T_END)
        sim = Simulator(cfg)
        res = sim.run()
        label = (f"{r['cost_name']}: wi={r['wi_opt']:.2g}, "
                 f"fc={r['fc_opt']:.0f}Hz, M_p={r['M_p_pct']:.1f}%, "
                 f"t_s={r['t_s_ms']:.1f}ms")
        ax1.plot(res["t"] * 1e3, res["v_C"], color=c, linewidth=1.1, label=label)
        ax2.plot(res["t"] * 1e3, res["i_L"], color=c, linewidth=1.1)

    # Linie zadanej + zaznacz eventy
    u_ref0 = default_config().controller.u_ref
    ax1.axhline(u_ref0, color="k", linestyle="--", linewidth=0.8, alpha=0.5,
                label=f"u_ref = {u_ref0} V")
    if SCENARIO is not None:
        if SCENARIO.ref_step_value is not None:
            ax1.axhline(SCENARIO.ref_step_value, color="grey", linestyle="--",
                        linewidth=0.8, alpha=0.5,
                        label=f"u_ref' = {SCENARIO.ref_step_value} V")
        if SCENARIO.load_step_time is not None:
            for ax in (ax1, ax2):
                ax.axvline(SCENARIO.load_step_time * 1e3, color="blue",
                           linestyle=":", linewidth=0.8, alpha=0.5)
        if SCENARIO.ref_step_time is not None:
            for ax in (ax1, ax2):
                ax.axvline(SCENARIO.ref_step_time * 1e3, color="red",
                           linestyle=":", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("$u_{dc}$ [V]")
    ax1.set_title("Porownanie przebiegow dla optymalnych nastaw z roznych funkcji celu")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("$i_L$ [A]")
    ax2.set_xlabel("czas [ms]")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out_path}")


def main() -> None:
    grid = load_grid()
    plot_cost_maps(grid)
    rows = optimal_table(grid)
    print_table(rows)
    plot_trajectories(rows)


if __name__ == "__main__":
    main()
