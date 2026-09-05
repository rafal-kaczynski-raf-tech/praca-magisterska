"""PSO od nowa (4 warianty) w nowym scenariuszu prof. Iwanskiego.

Scenariusz (optymalizacja/scenarios.py -> professor_scenario()):
  v_C0=100V, rozruch do u_ref=240V, wylacz/wlacz "urzadzenie" (R_load) w
  t=0.05/0.10/0.20/0.25 s, skok referencji 240->160V w t=0.15s, koniec t=0.30s.

Sterownik uzywa MODEL-BASED korekty pradowej (Eq.14-17, Tatari/Bizhani/
Iwanski) -- pso.CURRENT_CORRECTION = True. Dla kazdego z 4 wariantow funkcji
celu (ITAE / CurrentAware / CurrentOscillation / CurrentEffort) PSO szuka
nastaw (wi*, fc*) od nowa (nie recyklinguje starych npz z innych scenariuszy).

Wyniki zapisywane do optymalizacja/pso_prof_<wariant>.npz, uzywane potem przez
optymalizacja/analiza_scenariusz_prof.py.

Uruchomienie:  python -m optymalizacja.pso_professor_scenario
"""
from __future__ import annotations
import time
import numpy as np

import optymalizacja.pso as pso
from optymalizacja.scenarios import professor_scenario, PROF_T_END

# --- Podmiana aktywnego scenariusza + wlaczenie korekty pradowej ---
pso.SCENARIO = professor_scenario()
pso.T_END = PROF_T_END
pso.CURRENT_CORRECTION = True

VARIANTS = ["ITAE", "CurrentAware", "CurrentOscillation", "CurrentEffort"]

OUT_PATHS = {
    "ITAE": "optymalizacja/pso_prof_itae.npz",
    "CurrentAware": "optymalizacja/pso_prof_currentaware.npz",
    "CurrentOscillation": "optymalizacja/pso_prof_currentoscillation.npz",
    "CurrentEffort": "optymalizacja/pso_prof_currenteffort.npz",
}


def main() -> None:
    print("=" * 78)
    print("PSO w scenariuszu prof. Iwanskiego (4 warianty, current_correction=ON)")
    print("=" * 78)

    results = {}
    t_total = time.time()
    for name in VARIANTS:
        print(f"\n--- wariant: {name} ---")
        r = pso.run_pso(cost_name=name, verbose=True)
        results[name] = r
        np.savez(OUT_PATHS[name],
                  wi_opt=r["wi_opt"], fc_opt=r["fc_opt"], gbest_F=r["gbest_F"])
        print(f"  Zapisano: {OUT_PATHS[name]}")

    print(f"\nLaczny czas PSO: {time.time() - t_total:.1f}s")

    print("\n" + "=" * 78)
    print("PODSUMOWANIE NASTAW (scenariusz prof., current_correction=ON)")
    print("=" * 78)
    hdr = f"{'wariant':<20}{'wi*':>9}{'fc* [Hz]':>11}{'J*':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        print(f"{name:<20}{r['wi_opt']:>9.4f}{r['fc_opt']:>11.1f}{r['gbest_F']:>14.6g}")


if __name__ == "__main__":
    main()
