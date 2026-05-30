"""Porownanie PSO na WSZYSTKICH 5 funkcjach celu.

Dla kazdej funkcji celu uruchamia niezalezne PSO (ten sam seed dla powtarzalnosci),
zapisuje wyniki do pso_all_costs.npz, generuje:
  - tabele porownawcza w konsoli
  - pso_porownanie_zbieznosci.png: 5 krzywych zbieznosci na 1 wykresie
  - pso_porownanie_przebiegow.png: 5 trajektorii u_dc/i_L dla optimum kazdej J

Uruchomienie:
    python -m optymalizacja.pso_compare_all
"""
from __future__ import annotations
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.cost_functions import COST_FUNCTIONS
from optymalizacja.scenarios import hard_scenario, HARD_T_END
from optymalizacja.pso import run_pso

OUT_NPZ = "optymalizacja/pso_all_costs.npz"
SCENARIO = hard_scenario()
T_END = HARD_T_END


def main() -> None:
    results = {}
    t_total = time.time()
    for name in COST_FUNCTIONS.keys():
        print(f"\n{'#' * 70}")
        print(f"# PSO  ->  cost = {name}")
        print(f"{'#' * 70}")
        r = run_pso(cost_name=name, verbose=True)
        results[name] = r

    print(f"\nLaczny czas: {time.time() - t_total:.1f}s")

    # ------- TABELA -------
    print("\n" + "=" * 105)
    print("PORÓWNANIE WYNIKÓW PSO DLA 5 FUNKCJI CELU (scenariusz trudny: load OFF + ref UP)")
    print("=" * 105)
    headers = ["Cost", "wi*", "fc* [Hz]", "J*", "M_p [%]", "t_s [ms]",
               "e_ss [V]", "ripple [V]", "iL_peak [A]", "n_switch"]
    fmt = "{:<11} {:>7} {:>9} {:>12} {:>8} {:>9} {:>8} {:>10} {:>11} {:>9}"
    print(fmt.format(*headers))
    print("-" * 105)
    for name, r in results.items():
        m = r["metrics"]
        print(fmt.format(
            name,
            f"{r['wi_opt']:.3f}",
            f"{r['fc_opt']:.0f}",
            f"{r['gbest_F']:.4g}",
            f"{m['M_p_pct']:.2f}",
            f"{m['t_s_ms']:.2f}",
            f"{m['e_ss_V']:.3f}",
            f"{m['ripple_V']:.3f}",
            f"{m['iL_peak_A']:.2f}",
            f"{int(m['n_switch'])}",
        ))
    print("=" * 105)

    # ------- ZAPIS NPZ -------
    save_kwargs = {}
    for name, r in results.items():
        save_kwargs[f"{name}_history_F"] = r["history_gbest_F"]
        save_kwargs[f"{name}_history_X"] = r["history_gbest_X"]
        save_kwargs[f"{name}_wi_opt"] = r["wi_opt"]
        save_kwargs[f"{name}_fc_opt"] = r["fc_opt"]
        save_kwargs[f"{name}_gbest_F"] = r["gbest_F"]
    np.savez(OUT_NPZ, **save_kwargs)
    print(f"\nZapisano: {OUT_NPZ}")

    # ------- WYKRES ZBIEZNOSCI -------
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in results.items():
        F = r["history_gbest_F"]
        iters = np.arange(len(F))
        ax.plot(iters, F / F[0], "o-", linewidth=1.4, markersize=3, label=name)
    ax.set_xlabel("iteracja")
    ax.set_ylabel(r"$J_{best}(it)\,/\,J_{best}(0)$  (normalizowane)")
    ax.set_yscale("log")
    ax.set_title("Zbieznosc PSO dla 5 funkcji celu (normalizowane do startu)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out1 = "optymalizacja/pso_porownanie_zbieznosci.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out1}")

    # ------- WYKRES PRZEBIEGOW -------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, r), c in zip(results.items(), colors):
        base = default_config()
        cfg = replace(base,
                      controller=replace(base.controller,
                                          wi=r["wi_opt"], fc_lpf=r["fc_opt"]),
                      scenario=SCENARIO, T_end=T_END)
        res = Simulator(cfg).run()
        m = r["metrics"]
        label = (f"{name}: wi={r['wi_opt']:.2f}, fc={r['fc_opt']:.0f}Hz | "
                 f"M_p={m['M_p_pct']:.2f}%, t_s={m['t_s_ms']:.2f}ms, "
                 f"e_ss={m['e_ss_V']:.3f}V")
        ax1.plot(res["t"] * 1e3, res["v_C"], color=c, linewidth=1.0, label=label)
        ax2.plot(res["t"] * 1e3, res["i_L"], color=c, linewidth=1.0)

    ax1.axhline(240, ls="--", color="k", alpha=0.4, linewidth=0.7)
    if SCENARIO is not None and SCENARIO.ref_step_value is not None:
        ax1.axhline(SCENARIO.ref_step_value, ls="--", color="gray", alpha=0.4, linewidth=0.7)
    for ax in (ax1, ax2):
        if SCENARIO is not None and SCENARIO.load_step_time is not None:
            ax.axvline(SCENARIO.load_step_time * 1e3, ls=":", color="blue", alpha=0.4)
        if SCENARIO is not None and SCENARIO.ref_step_time is not None:
            ax.axvline(SCENARIO.ref_step_time * 1e3, ls=":", color="red", alpha=0.4)
    ax2.axhline(20, ls="--", color="r", alpha=0.4, linewidth=0.7)

    ax1.set_ylabel("$u_{dc}$ [V]")
    ax1.set_title("Przebiegi dla optimum PSO dla 5 funkcji celu (scenariusz trudny)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_ylabel("$i_L$ [A]")
    ax2.set_xlabel("czas [ms]")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out2 = "optymalizacja/pso_porownanie_przebiegow.png"
    fig.savefig(out2, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Zapisano: {out2}")


if __name__ == "__main__":
    main()
