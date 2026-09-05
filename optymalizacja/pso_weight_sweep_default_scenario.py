"""Powtorka calego sweepu wag (lam/mu/gamma) na scenariuszu Z WALIDACJI PSIM
(Krok 2.5/2.6) zamiast scenariusza stress -- na prosbe uzytkownika, po tym
jak profesor napisal, ze "w symulacji tetnienia byly mniejsze".

Scenariusz = default_scenario() (None): pojedynczy rozruch 100 -> 240 V,
BEZ powtarzanych skokow referencji, T_end = 100 ms (jak w oryginalnej
walidacji vs PSIM). Okno stanu ustalonego do pomiaru tetnien = [80, 100] ms
(dokladnie jak w porownanie_oop_vs_psim.py / psim_validation.md Krok 2.6).

Dla kazdego z 3 wariantow sweep wagi (wlasna baza x factor):
    CurrentAware:       lam,   baza LAMBDA_I=1.0
    CurrentOscillation: mu,    baza MU_OSC=0.02
    CurrentEffort:      gamma, baza GAMMA_EFFORT=0.02
Factory: 0.1, 0.3, 0.5, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0 (caly zestaw w jednym
przebiegu, zeby nie uruchamiac PSO dwa razy dla factor=1.0).

Uruchomienie:  python -m optymalizacja.pso_weight_sweep_default_scenario
"""
from __future__ import annotations
import numpy as np

import optymalizacja.pso as pso
from optymalizacja.scenarios import default_scenario
from optymalizacja.cost_functions import LAMBDA_I, MU_OSC, GAMMA_EFFORT
from src.config import default_config
from src.simulator import Simulator

# --- Podmiana aktywnego scenariusza na DEFAULT (jak w walidacji PSIM) ---
SCN = default_scenario()          # None
T_END = 0.10                      # s -- jak default_config()
pso.SCENARIO = SCN
pso.T_END = T_END

FACTORS = [0.1, 0.3, 0.5, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0]

VARIANTS = [
    ("Wariant 1 (uchyb pradu)", "CurrentAware", LAMBDA_I, "lam"),
    ("Wariant 2 (oscylacje 4 probek)", "CurrentOscillation", MU_OSC, "mu"),
    ("Wariant 3 (wielkosc pradu)", "CurrentEffort", GAMMA_EFFORT, "gamma"),
]

# Okno stanu ustalonego -- jak w Kroku 2.6 (porownanie_oop_vs_psim.py)
STEADY_T0, STEADY_T1 = 0.080, 0.100


def simulate(wi: float, fc: float):
    base = default_config()
    from dataclasses import replace
    cfg = replace(base, controller=replace(base.controller, wi=wi, fc_lpf=fc),
                  scenario=SCN, T_end=T_END)
    return Simulator(cfg).run(), base


def analyse_steady(res: dict, base) -> dict:
    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]
    u_ref = base.controller.u_ref

    m = (t >= STEADY_T0) & (t < STEADY_T1)
    iae_u = float(np.trapezoid(np.abs(v - u_ref), t))   # IAE napiecia (caly horyzont)
    return {
        "iae_u": iae_u,
        "iL_ripple_pp": float(np.ptp(iL[m])),
        "v_ripple_pp": float(np.ptp(v[m])),
        "iL_peak": float(np.max(iL)),
        "iL_mean_steady": float(np.mean(iL[m])),
    }


def main() -> None:
    for label, cost_name, base_weight, weight_name in VARIANTS:
        print("=" * 78)
        print(f"SWEEP WAGI {weight_name} ({label}) -- scenariusz WALIDACJI PSIM "
              f"(brak stress), baza {weight_name}={base_weight}")
        print("=" * 78)

        rows = []
        for factor in FACTORS:
            w = base_weight * factor
            print(f"\n--- {weight_name} = {w:.5f} ({factor*100:.0f}% bazy) ---")
            r = pso.run_pso(cost_name=cost_name, weight=w, verbose=False)
            wi, fc = r["wi_opt"], r["fc_opt"]
            res, base = simulate(wi, fc)
            a = analyse_steady(res, base)
            rows.append((factor, w, wi, fc, a["iae_u"], a["iL_ripple_pp"],
                         a["v_ripple_pp"], a["iL_peak"], a["iL_mean_steady"]))
            tag = f"{factor}".replace(".", "p")
            np.savez(f"optymalizacja/pso_default_{cost_name.lower()}_{tag}.npz",
                     wi_opt=wi, fc_opt=fc, gbest_F=r["gbest_F"], weight=w)
            print(f"  wi*={wi:.4f}  fc*={fc:.1f} Hz  IAE_u={a['iae_u']:.4f} V*s  "
                  f"iL_pp={a['iL_ripple_pp']:.3f} A  vC_pp={a['v_ripple_pp']:.4f} V  "
                  f"iL_peak={a['iL_peak']:.2f} A  iL_mean={a['iL_mean_steady']:.2f} A")

        print("\n" + "-" * 78)
        print(f"PODSUMOWANIE -- {label} (scenariusz walidacji PSIM)")
        print("-" * 78)
        hdr = f"{'factor':>7}{weight_name:>10}{'wi*':>9}{'fc* [Hz]':>11}" \
              f"{'IAE_u':>10}{'iL_pp [A]':>12}{'vC_pp [V]':>12}{'iL_peak [A]':>13}{'iL_mean [A]':>13}"
        print(hdr)
        print("-" * len(hdr))
        for factor, w, wi, fc, iae_u, il_pp, vc_pp, il_peak, il_mean in rows:
            print(f"{factor:>7.2f}{w:>10.5f}{wi:>9.4f}{fc:>11.1f}{iae_u:>10.4f}"
                  f"{il_pp:>12.3f}{vc_pp:>12.4f}{il_peak:>13.2f}{il_mean:>13.2f}")
        print()


if __name__ == "__main__":
    main()
