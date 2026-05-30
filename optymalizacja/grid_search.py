"""Grid search 2D nad (wi, fc_lpf) dla sterownika MF-BB.

Uruchamia symulator dla kazdej kombinacji parametrow, oblicza wszystkie
funkcje celu i klasyczne metryki, zapisuje wynik do pliku .npz dla
nastepnego kroku (analyze_results.py).

Uruchomienie:
    python -m optymalizacja.grid_search
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import replace

from src.config import default_config, ControllerParams
from src.simulator import Simulator
from optymalizacja.cost_functions import COST_FUNCTIONS
from optymalizacja import metrics
from optymalizacja.scenarios import hard_scenario, HARD_T_END


# Zakresy gridu (log-spaced dla obu osi)
WI_RANGE = np.logspace(np.log10(0.1), np.log10(5.0), 15)
FC_RANGE = np.logspace(np.log10(200.0), np.log10(10000.0), 15)

# Aktywny scenariusz testowy (None = default monotoniczny)
SCENARIO = hard_scenario()
T_END = HARD_T_END


def _build_u_ref_arr(t: np.ndarray, u_ref0: float, scn) -> np.ndarray:
    """Zwraca trajektorie wartosci zadanej u_ref(t) zgodna ze scenariuszem."""
    u_ref_arr = np.full_like(t, u_ref0)
    if scn is not None and scn.ref_step_time is not None:
        mask = t >= scn.ref_step_time
        u_ref_arr[mask] = scn.ref_step_value
    return u_ref_arr


def run_one(wi: float, fc_lpf: float) -> dict:
    """Pojedyncza symulacja dla zadanych (wi, fc_lpf). Zwraca slownik:
    {cost_name: value, metric_name: value}.
    """
    base_cfg = default_config()
    new_ctrl = replace(base_cfg.controller, wi=wi, fc_lpf=fc_lpf)
    cfg = replace(base_cfg, controller=new_ctrl, scenario=SCENARIO, T_end=T_END)

    sim = Simulator(cfg)
    res = sim.run()
    t = res["t"]
    u_ref0 = cfg.controller.u_ref
    u_ref_arr = _build_u_ref_arr(t, u_ref0, SCENARIO)
    u_ref_final = float(u_ref_arr[-1])

    record: dict = {}

    # Funkcje celu (na PELNYM przebiegu, z dynamicznym u_ref)
    for name, fn in COST_FUNCTIONS.items():
        record[f"cost_{name}"] = fn(t, res["v_C"], res["i_L"], res["s"], u_ref_arr)

    # Klasyczne metryki - liczone na segmencie PO OSTATNIM ZDARZENIU
    # (przedstawiaja "ostateczna jakosc" regulacji wzgledem finalnego setpointu)
    if SCENARIO is not None:
        event_times = [tev for tev in (SCENARIO.load_step_time, SCENARIO.ref_step_time)
                       if tev is not None]
        t_start_metrics = max(event_times) if event_times else 0.0
    else:
        t_start_metrics = 0.0
    seg = t >= t_start_metrics
    # Settling time mierzony od momentu ostatniego zdarzenia (przesuniecie czasu do 0)
    t_seg = t[seg] - t_start_metrics
    record.update(metrics.summary(t_seg, res["v_C"][seg], res["i_L"][seg],
                                   res["s"][seg], u_ref_final))
    return record


def main() -> None:
    n_wi = len(WI_RANGE)
    n_fc = len(FC_RANGE)
    total = n_wi * n_fc
    print(f"Grid search 2D: {n_wi} x {n_fc} = {total} symulacji")
    print(f"  wi  in [{WI_RANGE[0]:.3g}, {WI_RANGE[-1]:.3g}]  (log)")
    print(f"  fc  in [{FC_RANGE[0]:.0f}, {FC_RANGE[-1]:.0f}] Hz  (log)")

    # Bufory wynikow (jeden 2D array na kazdy metryke / koszt)
    results: dict = {}
    sample = run_one(WI_RANGE[0], FC_RANGE[0])
    for key in sample.keys():
        results[key] = np.zeros((n_wi, n_fc))
    for key, val in sample.items():
        results[key][0, 0] = val

    t0 = time.time()
    for i, wi in enumerate(WI_RANGE):
        for j, fc in enumerate(FC_RANGE):
            if i == 0 and j == 0:
                continue
            rec = run_one(wi, fc)
            for key, val in rec.items():
                results[key][i, j] = val
            done = i * n_fc + j + 1
            if done % 25 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done:3d}/{total}] elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s")

    elapsed = time.time() - t0
    print(f"\nGrid zakonczony w {elapsed:.1f}s")

    # Zapis do .npz
    out_path = "optymalizacja/grid_results.npz"
    np.savez(out_path, wi_range=WI_RANGE, fc_range=FC_RANGE, **results)
    print(f"Zapisano: {out_path}")


if __name__ == "__main__":
    main()
