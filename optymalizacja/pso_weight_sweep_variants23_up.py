"""Reczny sweep wagi W GORE dla Wariantu 2 (CurrentOscillation, mu) i
Wariantu 3 (CurrentEffort, gamma) -- analogicznie do pso_lambda_sweep_up.py,
kontynuacja pso_weight_sweep_variants23.py (sweep w dol nie potwierdzil
kierunku sugerowanego przez prof. Iwanskiego).

Dla kazdego wariantu sweep robiony jest wzgledem WLASNEGO bazowego ustawienia
(MU_OSC=0.02, GAMMA_EFFORT=0.02), skalowanego tymi samymi wspolczynnikami co
w Wariancie 1: 1.1 / 1.3 / 1.5 / 2.0 / 3.0.

Uruchomienie:  python -m optymalizacja.pso_weight_sweep_variants23_up
"""
from __future__ import annotations
import numpy as np

import optymalizacja.pso as pso
from optymalizacja.scenarios import stress_scenario, STRESS_T_END
from optymalizacja.analiza_ripple import simulate, analyse
from optymalizacja.cost_functions import MU_OSC, GAMMA_EFFORT
from src.config import default_config

# --- Podmiana aktywnego scenariusza na STRESS ---
SCN = stress_scenario()
pso.SCENARIO = SCN
pso.T_END = STRESS_T_END

FACTORS = [1.1, 1.3, 1.5, 2.0, 3.0]

VARIANTS = [
    ("Wariant 2 (oscylacje 4 probek)", "CurrentOscillation", MU_OSC, "mu"),
    ("Wariant 3 (wielkosc pradu)", "CurrentEffort", GAMMA_EFFORT, "gamma"),
]


def main() -> None:
    base = default_config()
    u_ref0 = base.controller.u_ref
    i_max = base.controller.i_max

    for label, cost_name, base_weight, weight_name in VARIANTS:
        print("=" * 78)
        print(f"SWEEP WAGI {weight_name} W GORE ({label}) -- stress scenario, "
              f"baza {weight_name}={base_weight}")
        print("=" * 78)

        rows = []
        for factor in FACTORS:
            w = base_weight * factor
            print(f"\n--- {weight_name} = {w:.5f} ({factor*100:.0f}% bazy) ---")
            r = pso.run_pso(cost_name=cost_name, weight=w, verbose=False)
            wi, fc = r["wi_opt"], r["fc_opt"]
            res = simulate(wi, fc)
            a = analyse(res, u_ref0, i_max)
            rows.append((factor, w, wi, fc, a["iae_u"], a["iL_ripple_pp"],
                         a["v_ripple_pp"], a["iL_peak"]))
            tag = f"{factor}".replace(".", "p")
            np.savez(f"optymalizacja/pso_{cost_name.lower()}_{tag}.npz",
                     wi_opt=wi, fc_opt=fc, gbest_F=r["gbest_F"], weight=w)
            print(f"  wi*={wi:.4f}  fc*={fc:.1f} Hz  IAE_u={a['iae_u']:.4f} V*s  "
                  f"iL_pp={a['iL_ripple_pp']:.3f} A  vC_pp={a['v_ripple_pp']:.4f} V  "
                  f"iL_peak={a['iL_peak']:.2f} A")

        print("\n" + "-" * 78)
        print(f"PODSUMOWANIE -- {label}")
        print("-" * 78)
        hdr = f"{'factor':>7}{weight_name:>10}{'wi*':>9}{'fc* [Hz]':>11}" \
              f"{'IAE_u':>10}{'iL_pp [A]':>12}{'vC_pp [V]':>12}{'iL_peak [A]':>13}"
        print(hdr)
        print("-" * len(hdr))
        for factor, w, wi, fc, iae_u, il_pp, vc_pp, il_peak in rows:
            print(f"{factor:>7.2f}{w:>10.5f}{wi:>9.4f}{fc:>11.1f}{iae_u:>10.4f}"
                  f"{il_pp:>12.3f}{vc_pp:>12.4f}{il_peak:>13.2f}")
        print()


if __name__ == "__main__":
    main()
