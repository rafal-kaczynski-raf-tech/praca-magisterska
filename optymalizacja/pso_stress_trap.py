"""Pelne PSO dla dwoch kryteriow na STRESS scenario (demonstracja "pulapki").

Scenariusz stress (powtarzalne skoki 240<->280 V co 20 ms) sprawia, ze
transienty pradu stanowia ~29% calego bledu pradu (vs ~5% w hard_scenario).
W takim scenariuszu kryterium swiadome pradu (CurrentAware) powinno przesunac
optimum ku WOLNIEJSZEJ odpowiedzi - "pulapka" prof. sie budzi.

Uruchamiamy PSO dwa razy na TYM SAMYM scenariuszu:
  - ITAE        : kryterium slepe na prad (baza)
  - CurrentAware: kryterium z kara za blad pradu (lam=1.0)

Porownujemy optima (wi, fc) i metryki fizyczne (iL_peak, czas narastania,
liczba przelaczen). Jesli CurrentAware wybiera nizsze iL_peak / wolniejszy
narost -> dowod, ze pulapka faktycznie spowalnia sterownik.
"""
from __future__ import annotations
import time
from dataclasses import replace
import numpy as np

import optymalizacja.pso as pso
from optymalizacja.scenarios import stress_scenario, STRESS_T_END
from src.config import default_config
from src.simulator import Simulator

# --- Podmiana aktywnego scenariusza na STRESS (run_pso uzywa modulowych) ---
SCN = stress_scenario()
pso.SCENARIO = SCN
pso.T_END = STRESS_T_END

# Pierwszy skok W GORE: 240 -> 280 V w t = 40 ms (z _STRESS_PULSES)
STEP_T = 0.040
STEP_TARGET = 280.0
STEP_WIN_END = 0.060      # nastepny skok (w dol) o 60 ms


def step_speed(wi: float, fc: float) -> dict:
    """Resymuluj optimum i zmierz CZYSTA szybkosc na pierwszym skoku 240->280.

    Zwraca czas narastania (do 99% = 277.2 V) liczony od momentu skoku oraz
    szczyt pradu i_L w oknie tego skoku. To metryka odporna na zmienna
    referencje (w przeciwienstwie do metrics.summary opartej o stale u_ref).
    """
    base = default_config()
    ctrl = replace(base.controller, wi=wi, fc_lpf=fc)
    cfg = replace(base, controller=ctrl, scenario=SCN, T_end=STRESS_T_END)
    res = Simulator(cfg).run()
    t, vC, iL = res["t"], res["v_C"], res["i_L"]
    win = (t >= STEP_T) & (t <= STEP_WIN_END)
    tw, vw, iw = t[win], vC[win], iL[win]
    thr = 0.99 * STEP_TARGET
    reached = np.where(vw >= thr)[0]
    t_rise = (tw[reached[0]] - STEP_T) * 1e3 if reached.size else float("nan")
    return {"t_rise_ms": t_rise, "iL_peak_step_A": float(iw.max()),
            "iL_peak_all_A": float(iL.max())}


def main() -> None:
    results = {}
    for cost_name in ("ITAE", "CurrentAware"):
        print(f"\n{'='*60}\nPSO: cost = {cost_name}  (STRESS scenario)\n{'='*60}")
        t0 = time.time()
        r = pso.run_pso(cost_name=cost_name, verbose=True)
        r["elapsed_s"] = time.time() - t0
        results[cost_name] = r
        np.savez(
            f"optymalizacja/pso_stress_{cost_name.lower()}.npz",
            wi_opt=r["wi_opt"], fc_opt=r["fc_opt"], gbest_F=r["gbest_F"],
            history_gbest_F=r["history_gbest_F"],
            history_gbest_X=r["history_gbest_X"], cost_name=r["cost_name"],
        )

    print(f"\n{'='*60}\nPOROWNANIE OPTIMOW (STRESS scenario)\n{'='*60}")
    hdr = f"{'kryterium':<14}{'wi*':>8}{'fc* [Hz]':>11}{'iL_peak':>10}" \
          f"{'t_rise[ms]':>12}{'n_switch':>10}"
    print(hdr)
    print("-" * len(hdr))
    speed = {}
    for name, r in results.items():
        sp = step_speed(r["wi_opt"], r["fc_opt"])
        speed[name] = sp
        print(f"{name:<14}{r['wi_opt']:>8.4f}{r['fc_opt']:>11.1f}"
              f"{sp['iL_peak_step_A']:>10.2f}{sp['t_rise_ms']:>12.3f}"
              f"{r['metrics']['n_switch']:>10d}")

    a, b = speed["ITAE"], speed["CurrentAware"]
    print(f"\nRoznica iL_peak (CurrentAware - ITAE) = "
          f"{b['iL_peak_step_A'] - a['iL_peak_step_A']:+.2f} A")
    print(f"Roznica t_rise  (CurrentAware - ITAE) = "
          f"{b['t_rise_ms'] - a['t_rise_ms']:+.3f} ms  "
          f"(dodatnie = WOLNIEJ -> pulapka dziala)")


if __name__ == "__main__":
    main()
