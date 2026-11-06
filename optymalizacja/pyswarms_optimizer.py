"""Moduł integracyjny łączący Bliźniaka Cyfrowego z biblioteką naukową PySwarms.

Stanowi oficjalny, zwalidowany interfejs optymalizacyjny dla procedury dwuetapowej 2D:
- KROK 1: GlobalBestPSO dobierający (w_i*, fc_i*) w stanie ustalonym.
- KROK 2: GlobalBestPSO dobierający (w_u*, fc_u*) w stanie przejściowym.

Wykorzystuje:
- pyswarms.single.GlobalBestPSO (Miranda, 2018)
- Liniowo malejącą wagę bezwładności w in [w_max, w_min] (Shi-Eberhart, 1999)
- Strategię brzegową odbicia lustrzanego (reflective boundary handler)
"""
from __future__ import annotations
import numpy as np
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO

from src.config import ConverterParams, TrendMFBBParams, SimulationConfig
from src.simulator import Simulator
from optymalizacja.scenarios import professor_scenario

CONV = ConverterParams(V_in=100.0, L=0.75e-3, C=200e-6, R_load=47.619, v_C0=100.0, i_L0=0.0)
DT_PHYS = 0.5e-6


def optimize_stage1_pyswarms(
    lam: float = 1.0,
    n_particles: int = 16,
    n_iter: int = 25,
    seed: int = 42,
    verbose: bool = False
) -> tuple[float, float, float]:
    """Krok 1: Dobór (w_i*, fc_i*) w stanie ustalonym pod obciążeniem 1.2 kW za pomocą PySwarms."""
    # Granice logarytmiczne: wi in [0.05, 2.0], fc_i in [100, 2000] Hz
    lb = np.array([np.log10(0.05), np.log10(100.0)])
    ub = np.array([np.log10(2.0),  np.log10(2000.0)])
    bounds = (lb, ub)

    def batch_eval(X: np.ndarray) -> np.ndarray:
        costs = []
        for x in X:
            wi = float(10.0 ** x[0])
            fc_i = float(10.0 ** x[1])
            ctrl = TrendMFBBParams(
                u_ref=240.0,
                T_s_ctrl=10e-6,
                wi=wi,
                wv=0.0,
                fc_i=fc_i,
                fc_u=321.0,
                i_max=20.0,
                i_min=-20.0,
            )
            cfg = SimulationConfig(converter=CONV, controller=ctrl, dt_phys=DT_PHYS, T_end=0.150, scenario=professor_scenario())
            res = Simulator(cfg).run()

            m = (res["t"] >= 0.120) & (res["t"] < 0.150)
            v = res["v_C"][m]
            iL = res["i_L"][m]
            ides = res["i_des_phys"][m]

            j_steady_u = float(np.mean(np.abs(v - 240.0)))
            j_steady_i = float(np.mean(np.abs(ides - iL)))
            costs.append(j_steady_u + lam * j_steady_i)
        return np.array(costs)

    options = {"c1": 1.5, "c2": 1.5, "w": 0.9}
    oh_strategy = {"w": "lin_variation"}

    rng = np.random.default_rng(seed)
    init_pos = rng.uniform(lb, ub, size=(n_particles, 2))
    init_pos[0] = np.array([np.log10(0.50), np.log10(321.0)])

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options=options,
        bounds=bounds,
        bh_strategy="reflective",
        oh_strategy=oh_strategy,
        init_pos=init_pos,
        ftol=-np.inf,
    )

    cost, pos = optimizer.optimize(batch_eval, iters=n_iter, verbose=verbose)
    best_wi = float(10.0 ** pos[0])
    best_fci = float(10.0 ** pos[1])
    return best_wi, best_fci, float(cost)


def optimize_stage2_pyswarms(
    wi_fixed: float,
    fci_fixed: float,
    w_trans_time: float = 0.05,
    n_particles: int = 16,
    n_iter: int = 25,
    seed: int = 43,
    verbose: bool = False
) -> tuple[float, float, float, float]:
    """Krok 2: Dobór (w_u*, fc_u*) w stanie przejściowym za pomocą PySwarms."""
    # Granice logarytmiczne: wu in [1.0, 35.0], fc_u in [100, 2000] Hz
    lb = np.array([np.log10(1.0),  np.log10(100.0)])
    ub = np.array([np.log10(35.0), np.log10(2000.0)])
    bounds = (lb, ub)

    def batch_eval(X: np.ndarray) -> np.ndarray:
        costs = []
        for x in X:
            wu = float(10.0 ** x[0])
            fc_u = float(10.0 ** x[1])
            ctrl = TrendMFBBParams(
                u_ref=240.0,
                T_s_ctrl=10e-6,
                wi=wi_fixed,
                wv=wu,
                fc_i=fci_fixed,
                fc_u=fc_u,
                i_max=20.0,
                i_min=-20.0,
            )
            cfg = SimulationConfig(converter=CONV, controller=ctrl, dt_phys=DT_PHYS, T_end=0.200, scenario=professor_scenario())
            res = Simulator(cfg).run()

            t = res["t"]
            v = res["v_C"]

            m150 = (t >= 0.150) & (t <= 0.155)
            raw_undershoot = max(0.0, 160.0 - float(np.min(v[m150])))
            undershoot_penalty = max(0.0, raw_undershoot - 0.2) ** 2

            m_reb = (t >= 0.1515) & (t <= 0.155)
            rebound = max(0.0, float(np.max(v[m_reb])) - 160.0)
            rebound_penalty = rebound ** 2

            m_trans = (t >= 0.150) & (t <= 0.160)
            j_trans_time = float(np.mean(np.abs(v[m_trans] - 160.0)))

            j_trans = undershoot_penalty + rebound_penalty + w_trans_time * j_trans_time
            costs.append(j_trans)
        return np.array(costs)

    options = {"c1": 1.5, "c2": 1.5, "w": 0.9}
    oh_strategy = {"w": "lin_variation"}

    rng = np.random.default_rng(seed)
    init_pos = rng.uniform(lb, ub, size=(n_particles, 2))
    init_pos[0] = np.array([np.log10(20.0), np.log10(321.0)])

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=2,
        options=options,
        bounds=bounds,
        bh_strategy="reflective",
        oh_strategy=oh_strategy,
        init_pos=init_pos,
        ftol=-np.inf,
    )

    cost, pos = optimizer.optimize(batch_eval, iters=n_iter, verbose=verbose)
    best_wu = float(10.0 ** pos[0])
    best_fcu = float(10.0 ** pos[1])

    # Ocena finalnego zapadu napięcia
    ctrl = TrendMFBBParams(
        u_ref=240.0,
        T_s_ctrl=10e-6,
        wi=wi_fixed,
        wv=best_wu,
        fc_i=fci_fixed,
        fc_u=best_fcu,
        i_max=20.0,
        i_min=-20.0,
    )
    res = Simulator(SimulationConfig(converter=CONV, controller=ctrl, dt_phys=DT_PHYS, T_end=0.200, scenario=professor_scenario())).run()
    m150 = (res["t"] >= 0.150) & (res["t"] <= 0.155)
    under_best = max(0.0, 160.0 - float(np.min(res["v_C"][m150])))

    return best_wu, best_fcu, under_best, float(cost)
