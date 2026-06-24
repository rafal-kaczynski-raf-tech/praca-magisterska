"""Analiza tetnien pradu/napiecia: ITAE (tylko napiecie) vs CurrentAware (war.1).

Odpowiedz na uwage prof. Iwanskiego: czy optimum dobrane TYLKO z uchybu
napiecia (ITAE) daje wieksze tetnienia pradu niz wariant 1 (oba uchyby)?
Dodatkowo rozbicie uchybu pradu |i_des - i_L| na czesc w stanie przejsciowym
(PULAPKA -- uklad celowo idzie na i_max, nie na i_des) i w stanie ustalonym.

Scenariusz: stress (8 skokow 240<->280 V co 20 ms).
Uruchomienie:  python -m optymalizacja.analiza_ripple
"""
from __future__ import annotations
import numpy as np
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.scenarios import stress_scenario, STRESS_T_END, _STRESS_PULSES


def simulate(wi: float, fc: float):
    """Symulacja na stress scenario dla zadanych nastaw (wi, fc)."""
    base = default_config()
    scn = stress_scenario()
    new_ctrl = replace(base.controller, wi=wi, fc_lpf=fc)
    cfg = replace(base, controller=new_ctrl, scenario=scn, T_end=STRESS_T_END)
    return Simulator(cfg).run()


def u_ref_array(t: np.ndarray, u_ref0: float) -> np.ndarray:
    arr = np.full_like(t, u_ref0)
    for t_p, u_p in _STRESS_PULSES:
        arr[t >= t_p] = u_p
    return arr


# Granice segmentow miedzy skokami referencji
STEP_TIMES = [tp for tp, _ in _STRESS_PULSES]          # [0.040, ..., 0.180]
SEG_EDGES = [0.0] + STEP_TIMES + [STRESS_T_END]        # 0 .. 0.20
TRANS_WIN = 3.0e-3     # 3 ms po kazdym skoku = stan przejsciowy


def analyse(res: dict, u_ref0: float, i_max: float) -> dict:
    t = res["t"]
    v = res["v_C"]
    iL = res["i_L"]
    i_des = res["i_des_phys"]
    u_ref = u_ref_array(t, u_ref0)

    # --- 1. Uchyb napiecia (caly horyzont) ---
    iae_u = float(np.trapezoid(np.abs(v - u_ref), t))

    # --- 2. Uchyb pradu |i_des - i_L|, rozbity na transient vs steady ---
    e_i = np.abs(i_des - iL)
    trans_mask = np.zeros_like(t, dtype=bool)
    for ts in [0.0] + STEP_TIMES:                       # 0 = rozruch tez transient
        trans_mask |= (t >= ts) & (t < ts + TRANS_WIN)
    steady_mask = ~trans_mask
    iae_i_total = float(np.trapezoid(e_i, t))
    iae_i_trans = float(np.trapezoid(np.where(trans_mask, e_i, 0.0), t))
    iae_i_steady = float(np.trapezoid(np.where(steady_mask, e_i, 0.0), t))

    # Ile czasu w transiencie iL siedzi na ograniczniku i_max (>=99% i_max)?
    on_imax = trans_mask & (iL >= 0.99 * i_max)
    frac_on_imax = float(on_imax.sum()) / float(max(1, trans_mask.sum()))

    # --- 3. Tetnienia w stanie USTALONYM (ostatnie 40% kazdego segmentu) ---
    ripple_iL_pp, ripple_iL_std, ripple_v_pp, ripple_v_std = [], [], [], []
    for a, b in zip(SEG_EDGES[:-1], SEG_EDGES[1:]):
        if a == 0.0:                                    # pomijamy rozruch
            continue
        dur = b - a
        w0 = a + 0.6 * dur                              # ostatnie 40% segmentu
        m = (t >= w0) & (t < b)
        if m.sum() < 5:
            continue
        ripple_iL_pp.append(float(np.ptp(iL[m])))
        ripple_iL_std.append(float(np.std(iL[m])))
        ripple_v_pp.append(float(np.ptp(v[m])))
        ripple_v_std.append(float(np.std(v[m])))

    return {
        "iae_u": iae_u,
        "iae_i_total": iae_i_total,
        "iae_i_trans": iae_i_trans,
        "iae_i_steady": iae_i_steady,
        "trans_share_pct": 100.0 * iae_i_trans / iae_i_total,
        "frac_on_imax_pct": 100.0 * frac_on_imax,
        "iL_ripple_pp": float(np.mean(ripple_iL_pp)),
        "iL_ripple_std": float(np.mean(ripple_iL_std)),
        "v_ripple_pp": float(np.mean(ripple_v_pp)),
        "v_ripple_std": float(np.mean(ripple_v_std)),
        "iL_peak": float(np.max(iL)),
    }


def main() -> None:
    base = default_config()
    u_ref0 = base.controller.u_ref          # 240 V
    i_max = base.controller.i_max           # 20 A

    cases = {
        "ITAE (tylko napiecie)": "optymalizacja/pso_stress_itae.npz",
        "CurrentAware (war.1)":  "optymalizacja/pso_stress_currentaware.npz",
    }

    print("=" * 78)
    print("ANALIZA TETNIEN -- stress scenario (8 skokow 240<->280 V)")
    print(f"i_max = {i_max:.0f} A,  okno transientu = {TRANS_WIN*1e3:.0f} ms po skoku")
    print("=" * 78)

    results = {}
    for label, path in cases.items():
        d = np.load(path)
        wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
        res = simulate(wi, fc)
        a = analyse(res, u_ref0, i_max)
        results[label] = (wi, fc, a)
        print(f"\n### {label}")
        print(f"  nastawy:                wi = {wi:.4f},  fc = {fc:.0f} Hz")
        print(f"  IAE napiecia:           {a['iae_u']:.4f} V*s")
        print(f"  IAE pradu (calosc):     {a['iae_i_total']:.4f} A*s")
        print(f"     - w transiencie:     {a['iae_i_trans']:.4f} A*s "
              f"({a['trans_share_pct']:.1f}% calego uchybu pradu)  <- PULAPKA")
        print(f"     - w stanie ustal.:   {a['iae_i_steady']:.4f} A*s")
        print(f"  iL na i_max w transienc.: {a['frac_on_imax_pct']:.1f}% czasu okna")
        print(f"  TETNIENIA PRADU iL  (stan ustalony):")
        print(f"     - peak-to-peak:      {a['iL_ripple_pp']:.4f} A")
        print(f"     - odch.std:          {a['iL_ripple_std']:.4f} A")
        print(f"  TETNIENIA NAPIECIA vC (stan ustalony):")
        print(f"     - peak-to-peak:      {a['v_ripple_pp']:.4f} V")
        print(f"     - odch.std:          {a['v_ripple_std']:.4f} V")
        print(f"  iL_peak (caly horyzont): {a['iL_peak']:.2f} A")

    # --- Porownanie wprost ---
    print("\n" + "=" * 78)
    print("POROWNANIE (CurrentAware wzgledem ITAE)")
    print("=" * 78)
    _, _, a0 = results["ITAE (tylko napiecie)"]
    _, _, a1 = results["CurrentAware (war.1)"]

    def cmp(name, k, unit, lower_better=True):
        d0, d1 = a0[k], a1[k]
        delta = d1 - d0
        rel = 100.0 * delta / d0 if d0 != 0 else float("nan")
        arrow = "lepiej" if (delta < 0) == lower_better else "gorzej"
        print(f"  {name:28s} {d0:9.4f} -> {d1:9.4f} {unit}  "
              f"({rel:+6.1f}%  {arrow})")

    cmp("IAE napiecia", "iae_u", "V*s")
    cmp("Tetnienia iL (p-p)", "iL_ripple_pp", "A")
    cmp("Tetnienia iL (std)", "iL_ripple_std", "A")
    cmp("Tetnienia vC (p-p)", "v_ripple_pp", "V")
    cmp("Tetnienia vC (std)", "v_ripple_std", "V")
    print("=" * 78)


if __name__ == "__main__":
    main()
