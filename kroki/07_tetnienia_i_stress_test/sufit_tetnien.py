"""Sufit tetnien: ile w ogole da sie zbic tetnienia pradu STROJENIEM (wi, fc)?

Skanuje siatke (wi, fc) na stress scenario i dla kazdego punktu liczy
tetnienia pradu w stanie ustalonym + uchyb napiecia. Szuka punktu o
MINIMALNYCH tetnieniach i raportuje, jakim kosztem napiecia.

Cel: rozstrzygnac, czy duze tetnienia to problem do rozwiazania strojeniem,
czy strukturalny (limit-cycle przelaczania / wartosc L).

Uruchomienie:  python -m optymalizacja.sufit_tetnien
"""
from __future__ import annotations
import numpy as np
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.scenarios import stress_scenario, STRESS_T_END, _STRESS_PULSES

STEP_TIMES = [tp for tp, _ in _STRESS_PULSES]
SEG_EDGES = [0.0] + STEP_TIMES + [STRESS_T_END]


def simulate(wi: float, fc: float):
    base = default_config()
    scn = stress_scenario()
    cfg = replace(base, controller=replace(base.controller, wi=wi, fc_lpf=fc),
                  scenario=scn, T_end=STRESS_T_END)
    return Simulator(cfg).run()


def u_ref_array(t, u_ref0):
    arr = np.full_like(t, u_ref0)
    for t_p, u_p in _STRESS_PULSES:
        arr[t >= t_p] = u_p
    return arr


def steady_metrics(res, u_ref0):
    """Zwraca (ripple_iL_pp, ripple_iL_pct_of_mean, iae_u)."""
    t, v, iL = res["t"], res["v_C"], res["i_L"]
    iae_u = float(np.trapezoid(np.abs(v - u_ref_array(t, u_ref0)), t))
    pp, mean = [], []
    for a, b in zip(SEG_EDGES[:-1], SEG_EDGES[1:]):
        if a == 0.0:
            continue
        w0 = a + 0.6 * (b - a)
        m = (t >= w0) & (t < b)
        if m.sum() < 5:
            continue
        pp.append(float(np.ptp(iL[m])))
        mean.append(float(np.mean(iL[m])))
    ripple_pp = float(np.mean(pp))
    ripple_pct = 100.0 * ripple_pp / float(np.mean(mean))
    return ripple_pp, ripple_pct, iae_u


def main():
    base = default_config()
    u_ref0 = base.controller.u_ref

    # Siatka 12x12 w tej samej przestrzeni log co PSO
    wi_grid = np.logspace(np.log10(0.1), np.log10(5.0), 12)
    fc_grid = np.logspace(np.log10(200.0), np.log10(10000.0), 12)

    print("=" * 74)
    print("SUFIT TETNIEN -- skan siatki 12x12 (wi, fc) na stress scenario")
    print("=" * 74)

    best_ripple = (1e9, None)
    best_volt = (1e9, None)
    rows = []
    for wi in wi_grid:
        for fc in fc_grid:
            res = simulate(wi, fc)
            pp, pct, iae_u = steady_metrics(res, u_ref0)
            rows.append((wi, fc, pp, pct, iae_u))
            if pp < best_ripple[0]:
                best_ripple = (pp, (wi, fc, pp, pct, iae_u))
            if iae_u < best_volt[0]:
                best_volt = (iae_u, (wi, fc, pp, pct, iae_u))

    pps = np.array([r[2] for r in rows])
    print(f"\nTetnienia pradu (p-p) w calej siatce:")
    print(f"   min = {pps.min():.3f} A   max = {pps.max():.3f} A   "
          f"mediana = {np.median(pps):.3f} A")

    wi, fc, pp, pct, iae_u = best_ripple[1]
    print(f"\n### Punkt o NAJMNIEJSZYCH tetnieniach (sufit strojenia):")
    print(f"   wi = {wi:.3f}, fc = {fc:.0f} Hz")
    print(f"   tetnienia iL  = {pp:.3f} A p-p  ({pct:.1f}% sredniego pradu)")
    print(f"   IAE napiecia  = {iae_u:.4f} V*s")

    wi, fc, pp, pct, iae_u = best_volt[1]
    print(f"\n### Punkt o NAJLEPSZYM napieciu (dla porownania):")
    print(f"   wi = {wi:.3f}, fc = {fc:.0f} Hz")
    print(f"   tetnienia iL  = {pp:.3f} A p-p  ({pct:.1f}% sredniego pradu)")
    print(f"   IAE napiecia  = {iae_u:.4f} V*s")

    # Znane optima dla kontekstu
    print(f"\n### Dla porownania -- optima z PSO:")
    for label, path in [("ITAE", "optymalizacja/pso_stress_itae.npz"),
                        ("CurrentAware", "optymalizacja/pso_stress_currentaware.npz")]:
        d = np.load(path)
        res = simulate(float(d["wi_opt"]), float(d["fc_opt"]))
        pp, pct, iae_u = steady_metrics(res, u_ref0)
        print(f"   {label:13s}: tetnienia = {pp:.3f} A p-p ({pct:.1f}%), "
              f"IAE_u = {iae_u:.4f}")
    print("=" * 74)


if __name__ == "__main__":
    main()
