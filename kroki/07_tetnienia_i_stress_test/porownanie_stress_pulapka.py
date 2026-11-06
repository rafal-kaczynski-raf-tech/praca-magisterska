"""Wykres demonstracji "pulapki" na STRESS scenario.

Wczytuje optima PSO (ITAE vs CurrentAware) z plikow .npz, resymuluje oba
sterowniki na stress scenario i rysuje:
  - gora: u_dc(t) obu sterownikow + schodkowa referencja (240<->280 V)
  - dol : i_L(t) obu sterownikow + i_des (zadany prad)
Zoom na pierwszy skok 240->280 V (t=40 ms) pokazuje, czy CurrentAware
narasta WOLNIEJ i z nizszym szczytem pradu - czyli czy pulapka dziala.
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.scenarios import stress_scenario, STRESS_T_END
from optymalizacja.pso import _build_u_ref_arr

SCN = stress_scenario()


def simulate(wi: float, fc: float):
    base = default_config()
    ctrl = replace(base.controller, wi=wi, fc_lpf=fc)
    cfg = replace(base, controller=ctrl, scenario=SCN, T_end=STRESS_T_END)
    return Simulator(cfg).run()


def load_opt(path: str):
    d = np.load(path)
    return float(d["wi_opt"]), float(d["fc_opt"]), float(d["gbest_F"])


def main() -> None:
    wi_i, fc_i, _ = load_opt("optymalizacja/pso_stress_itae.npz")
    wi_c, fc_c, _ = load_opt("optymalizacja/pso_stress_currentaware.npz")

    res_i = simulate(wi_i, fc_i)
    res_c = simulate(wi_c, fc_c)
    t = res_i["t"]
    u_ref = _build_u_ref_arr(t, default_config().controller.u_ref, SCN)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    # --- Pelny horyzont: napiecie ---
    ax[0, 0].plot(t * 1e3, u_ref, "k--", lw=1, label="u_ref (240<->280)")
    ax[0, 0].plot(t * 1e3, res_i["v_C"], "tab:gray",
                  label=f"ITAE (wi={wi_i:.3f}, fc={fc_i:.0f})")
    ax[0, 0].plot(t * 1e3, res_c["v_C"], "tab:blue",
                  label=f"CurrentAware (wi={wi_c:.3f}, fc={fc_c:.0f})")
    ax[0, 0].set_title("u_dc - pelny horyzont (stress)")
    ax[0, 0].set_xlabel("t [ms]"); ax[0, 0].set_ylabel("u_dc [V]")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

    # --- Pelny horyzont: prad ---
    ax[1, 0].plot(t * 1e3, res_i["i_des_phys"], "k:", lw=1, label="i_des")
    ax[1, 0].plot(t * 1e3, res_i["i_L"], "tab:gray", label="ITAE")
    ax[1, 0].plot(t * 1e3, res_c["i_L"], "tab:blue", label="CurrentAware")
    ax[1, 0].set_title("i_L - pelny horyzont (stress)")
    ax[1, 0].set_xlabel("t [ms]"); ax[1, 0].set_ylabel("i_L [A]")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # --- Zoom na pierwszy skok 240->280 (t=40 ms) ---
    z = (t >= 0.038) & (t <= 0.052)
    ax[0, 1].plot(t[z] * 1e3, u_ref[z], "k--", lw=1)
    ax[0, 1].plot(t[z] * 1e3, res_i["v_C"][z], "tab:gray", label="ITAE")
    ax[0, 1].plot(t[z] * 1e3, res_c["v_C"][z], "tab:blue", label="CurrentAware")
    ax[0, 1].set_title("ZOOM skok 240->280 V: u_dc")
    ax[0, 1].set_xlabel("t [ms]"); ax[0, 1].set_ylabel("u_dc [V]")
    ax[0, 1].legend(fontsize=8); ax[0, 1].grid(alpha=0.3)

    ax[1, 1].plot(t[z] * 1e3, res_i["i_L"][z], "tab:gray", label="ITAE")
    ax[1, 1].plot(t[z] * 1e3, res_c["i_L"][z], "tab:blue", label="CurrentAware")
    ax[1, 1].set_title("ZOOM skok 240->280 V: i_L (szczyt pradu)")
    ax[1, 1].set_xlabel("t [ms]"); ax[1, 1].set_ylabel("i_L [A]")
    ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

    fig.suptitle("Pulapka w akcji: ITAE vs CurrentAware na stress scenario",
                 fontsize=13)
    fig.tight_layout()
    out = "porownanie_stress_pulapka.png"
    fig.savefig(out, dpi=130)
    print(f"Zapisano {out}")


if __name__ == "__main__":
    main()
