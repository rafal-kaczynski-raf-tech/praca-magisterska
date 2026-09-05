"""Analiza scenariusza prof. Iwanskiego: uchyb napiecia + tetnienia pradu
wzgledem sredniej, TYLKO w stanie ustalonym (nie w chwili przeskoku).

Scenariusz (optymalizacja/scenarios.py -> professor_scenario()):
  v_C0=100V, rozruch do 240V, wylacz/wlacz "urzadzenie" (R_load) w
  t=0.05/0.10/0.20/0.25s, skok referencji 240->160V w t=0.15s, koniec t=0.30s.

Dla kazdego z 4 wariantow (ITAE / CurrentAware / CurrentOscillation /
CurrentEffort), z nastawami (wi*, fc*) wyznaczonymi przez
pso_professor_scenario.py (osobne PSO w TYM scenariuszu, current_correction
wlaczona - Eq.14-17), liczone sa w kazdym segmencie miedzy zdarzeniami
(pomijajac pierwszy segment = rozruch):
  1) uchyb napiecia w stanie ustalonym:  mean(v_C) - u_ref_segmentu  [V]
  2) tetnienia pradu wzgledem wartosci sredniej: ptp(i_L)/mean(i_L)*100 [%]
     (gdy |mean(i_L)| < 0.5A - segment "urzadzenie wylaczone" - wynik %
     nieinformacyjny, oznaczony jako N/A, podawana jest tylko wartosc pp[A])

Stan ustalony = ostatnie 40% kazdego segmentu (jak w analiza_ripple.py).

Uruchomienie:  python -m optymalizacja.analiza_scenariusz_prof
"""
from __future__ import annotations
import numpy as np
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.scenarios import professor_scenario, PROF_T_END

SEG_EDGES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, PROF_T_END]

CASES = {
    "ITAE":               "optymalizacja/pso_prof_itae.npz",
    "CurrentAware":       "optymalizacja/pso_prof_currentaware.npz",
    "CurrentOscillation": "optymalizacja/pso_prof_currentoscillation.npz",
    "CurrentEffort":      "optymalizacja/pso_prof_currenteffort.npz",
}


def simulate(wi: float, fc: float) -> dict:
    """Symulacja w scenariuszu prof., sterownik z wlaczona korekta pradowa."""
    base = default_config()
    scn = professor_scenario()
    new_ctrl = replace(base.controller, wi=wi, fc_lpf=fc, current_correction=True)
    cfg = replace(base, controller=new_ctrl, scenario=scn, T_end=PROF_T_END)
    return Simulator(cfg).run()


def u_ref_for_segment(a: float, u_ref0: float, ref_step_time: float,
                       ref_step_value: float) -> float:
    return ref_step_value if a >= ref_step_time - 1e-12 else u_ref0


def analyse(res: dict, u_ref0: float, ref_step_time: float,
            ref_step_value: float) -> list[dict]:
    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]

    rows = []
    for a, b in zip(SEG_EDGES[:-1], SEG_EDGES[1:]):
        if a == 0.0:                              # pomijamy rozruch
            continue
        dur = b - a
        w0 = a + 0.6 * dur                        # ostatnie 40% segmentu
        m = (t >= w0) & (t < b)
        if m.sum() < 5:
            continue
        u_ref_seg = u_ref_for_segment(a, u_ref0, ref_step_time, ref_step_value)
        v_w, iL_w = v[m], iL[m]
        iL_mean = float(np.mean(iL_w))
        iL_pp = float(np.ptp(iL_w))
        iL_ripple_pct = (100.0 * iL_pp / abs(iL_mean)
                          if abs(iL_mean) >= 0.5 else float("nan"))
        rows.append({
            "seg": f"[{a:.2f}, {b:.2f})",
            "u_ref_V": u_ref_seg,
            "v_err_V": float(np.mean(v_w) - u_ref_seg),
            "iL_mean_A": iL_mean,
            "iL_ripple_pp_A": iL_pp,
            "iL_ripple_pct": iL_ripple_pct,
        })
    return rows


def main() -> None:
    base = default_config()
    u_ref0 = base.controller.u_ref            # 240 V
    scn = professor_scenario()

    print("=" * 92)
    print("ANALIZA SCENARIUSZA PROF. IWANSKIEGO (stan ustalony, current_correction=ON)")
    print("=" * 92)

    for name, path in CASES.items():
        try:
            d = np.load(path)
        except FileNotFoundError:
            print(f"\n### {name}: BRAK {path} - uruchom najpierw "
                  f"pso_professor_scenario.py")
            continue
        wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
        res = simulate(wi, fc)
        rows = analyse(res, u_ref0, scn.ref_step_time, scn.ref_step_value)

        print(f"\n### {name}   (wi*={wi:.4f}, fc*={fc:.0f} Hz)")
        hdr = f"{'segment [s]':<14}{'u_ref [V]':>10}{'v_err [V]':>11}" \
              f"{'iL_mean [A]':>12}{'iL_pp [A]':>11}{'iL_ripple [%]':>14}"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            pct_str = (f"{r['iL_ripple_pct']:.2f}"
                       if not np.isnan(r["iL_ripple_pct"]) else "N/A")
            print(f"{r['seg']:<14}{r['u_ref_V']:>10.1f}{r['v_err_V']:>11.4f}"
                  f"{r['iL_mean_A']:>12.4f}{r['iL_ripple_pp_A']:>11.4f}"
                  f"{pct_str:>14}")


if __name__ == "__main__":
    main()
