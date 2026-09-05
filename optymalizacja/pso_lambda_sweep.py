"""Reczny sweep wagi czlonu pradowego lam w wariancie 1 (CurrentAware), na
prosbe prof. Iwanskiego: "niech Pan sprobuje recznie zmieniac wage dla bledu
pradu i zobaczyc wyniki, np. 0.1 - bo w symulacji te tetnienia pradu w stanie
ustalonym byly mniejsze niz to co Pan dostaje".

Uruchamia PELNE PSO (ten sam stress scenario) dla lam w {1.0 (obecne), 0.5,
0.3, 0.1} i porownuje: nastawy (wi*, fc*), IAE napiecia, tetnienia i_L/v_C
(peak-to-peak, metoda analiza_ripple.analyse -- usredniona po 8 skokach) oraz
szczyt pradu.

Uruchomienie:  python -m optymalizacja.pso_lambda_sweep
"""
from __future__ import annotations
import numpy as np

import optymalizacja.pso as pso
from optymalizacja.scenarios import stress_scenario, STRESS_T_END
from optymalizacja.analiza_ripple import simulate, analyse
from src.config import default_config

# --- Podmiana aktywnego scenariusza na STRESS (jak w pso_stress_trap.py) ---
SCN = stress_scenario()
pso.SCENARIO = SCN
pso.T_END = STRESS_T_END

LAMBDAS = [1.0, 0.5, 0.3, 0.1]


def main() -> None:
    base = default_config()
    u_ref0 = base.controller.u_ref
    i_max = base.controller.i_max

    print("=" * 78)
    print("SWEEP WAGI lam (CurrentAware, wariant 1) -- stress scenario")
    print("=" * 78)

    rows = []
    for lam in LAMBDAS:
        print(f"\n--- lam = {lam} ---")
        r = pso.run_pso(cost_name="CurrentAware", weight=lam, verbose=False)
        wi, fc = r["wi_opt"], r["fc_opt"]
        res = simulate(wi, fc)
        a = analyse(res, u_ref0, i_max)
        rows.append((lam, wi, fc, a["iae_u"], a["iL_ripple_pp"],
                     a["v_ripple_pp"], a["iL_peak"]))
        np.savez(f"optymalizacja/pso_lambda_{str(lam).replace('.', 'p')}.npz",
                 wi_opt=wi, fc_opt=fc, gbest_F=r["gbest_F"], lam=lam)
        print(f"  wi*={wi:.4f}  fc*={fc:.1f} Hz  IAE_u={a['iae_u']:.4f} V*s  "
              f"iL_pp={a['iL_ripple_pp']:.3f} A  vC_pp={a['v_ripple_pp']:.4f} V  "
              f"iL_peak={a['iL_peak']:.2f} A")

    print("\n" + "=" * 78)
    print("PODSUMOWANIE")
    print("=" * 78)
    hdr = f"{'lam':>6}{'wi*':>9}{'fc* [Hz]':>11}{'IAE_u':>10}" \
          f"{'iL_pp [A]':>12}{'vC_pp [V]':>12}{'iL_peak [A]':>13}"
    print(hdr)
    print("-" * len(hdr))
    for lam, wi, fc, iae_u, il_pp, vc_pp, il_peak in rows:
        print(f"{lam:>6.2f}{wi:>9.4f}{fc:>11.1f}{iae_u:>10.4f}"
              f"{il_pp:>12.3f}{vc_pp:>12.4f}{il_peak:>13.2f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
