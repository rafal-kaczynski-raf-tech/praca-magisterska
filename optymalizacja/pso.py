"""Particle Swarm Optimization dla nastaw sterownika MF-BB.

Implementacja kanonicznego PSO (Kennedy & Eberhart 1995, Shi & Eberhart 1998)
z liniowo malejacym wagiem bezwladnosci. Optymalizacja prowadzona w
przestrzeni logarytmicznej (log10 wi, log10 fc_lpf) -- zarowno wi jak fc_lpf
zmieniaja sie o rzedy wielkosci, wiec parametry log-uniform pasuja do
charakterystyki problemu.

Algorytm:
    1. Inicjalizacja N czastek losowo w (log_wi_min, log_wi_max) x (log_fc_min, log_fc_max)
    2. Dla kazdej iteracji:
       a) ocena funkcji celu (rownolegle czastka po czastce)
       b) update best personal (pbest) i best global (gbest)
       c) update predkosci:
            v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
       d) update pozycji:
            x <- x + v   (z odbiciem od granic)
    3. w maleje liniowo od w_max do w_min -- exploration -> exploitation

References:
    - Kennedy J., Eberhart R., "Particle Swarm Optimization", IEEE ICNN 1995.
    - Shi Y., Eberhart R., "A modified particle swarm optimizer", IEEE 1998
      (inertia weight).

Uruchomienie:
    python -m optymalizacja.pso
"""
from __future__ import annotations
import time
import numpy as np
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.cost_functions import (COST_FUNCTIONS, CURRENT_AWARE,
                                           OSCILLATION_AWARE, CURRENT_EFFORT)
from optymalizacja import metrics
from optymalizacja.scenarios import hard_scenario, HARD_T_END


# Konfiguracja PSO
COST_NAME = "ITAE"          # wybrana funkcja celu (najwiekszy spread w gridzie)
N_PARTICLES = 20
N_ITER = 40
W_MAX, W_MIN = 0.9, 0.4     # waga bezwladnosci (linearnie malejaca)
C1 = 2.05                   # wspolczynnik kognitywny
C2 = 2.05                   # wspolczynnik spoleczny
V_FRAC = 0.5                # maksymalna predkosc = V_FRAC * (ub-lb)

# Granice w przestrzeni logarytmicznej (zgodne z gridem 15x15)
LOG_WI_BOUNDS = (np.log10(0.1), np.log10(5.0))
LOG_FC_BOUNDS = (np.log10(200.0), np.log10(10000.0))

# Aktywny scenariusz
SCENARIO = hard_scenario()
T_END = HARD_T_END

# Model-based current correction (Eq.14-17, Tatari/Bizhani/Iwanski) - domyslnie
# wylaczona (stara funkcja przelaczajaca). Ustaw pso.CURRENT_CORRECTION = True
# przed wywolaniem run_pso(), zeby wlaczyc korekte w ewaluowanym sterowniku.
CURRENT_CORRECTION = False

RNG_SEED = 42


def _build_u_ref_arr(t: np.ndarray, u_ref0: float, scn) -> np.ndarray:
    u_ref_arr = np.full_like(t, u_ref0)
    if scn is not None and scn.ref_step_time is not None:
        mask = t >= scn.ref_step_time
        u_ref_arr[mask] = scn.ref_step_value
    if scn is not None and getattr(scn, "ref_pulses", None) is not None:
        for t_p, u_p in scn.ref_pulses:
            u_ref_arr[t >= t_p] = u_p
    return u_ref_arr


def evaluate(log_wi: float, log_fc: float, cost_name: str = COST_NAME,
             weight: float | None = None) -> float:
    """Pojedyncza ewaluacja funkcji celu (z przestrzeni log -> param fizyczny).

    weight: opcjonalne nadpisanie wagi czlonu pradowego -- dla CurrentAware to
    'lam', dla CurrentOscillation to 'mu', dla CurrentEffort to 'gamma'.
    Domyslnie None -> uzywa stalej modulowej z cost_functions.py. Sluzy do
    recznego sweepu wagi na prosbe prof. Iwanskiego.
    """
    wi = 10.0 ** log_wi
    fc = 10.0 ** log_fc

    base = default_config()
    new_ctrl = replace(base.controller, wi=wi, fc_lpf=fc,
                        current_correction=CURRENT_CORRECTION)
    cfg = replace(base, controller=new_ctrl, scenario=SCENARIO, T_end=T_END)

    sim = Simulator(cfg)
    res = sim.run()
    t = res["t"]
    u_ref_arr = _build_u_ref_arr(t, cfg.controller.u_ref, SCENARIO)

    fn = COST_FUNCTIONS[cost_name]
    if cost_name in CURRENT_AWARE:
        kwargs = {"i_des": res["i_des_phys"]}
        if weight is not None:
            kwargs["lam"] = weight
        return fn(t, res["v_C"], res["i_L"], res["s"], u_ref_arr, **kwargs)
    if cost_name in OSCILLATION_AWARE:
        kwargs = {"iL_ctrl": res["iL_sample"]}
        if weight is not None:
            kwargs["mu"] = weight
        return fn(t, res["v_C"], res["i_L"], res["s"], u_ref_arr, **kwargs)
    if cost_name in CURRENT_EFFORT:
        kwargs = {}
        if weight is not None:
            kwargs["gamma"] = weight
        return fn(t, res["v_C"], res["i_L"], res["s"], u_ref_arr, **kwargs)
    return fn(t, res["v_C"], res["i_L"], res["s"], u_ref_arr)


def run_pso(n_particles: int = N_PARTICLES,
            n_iter: int = N_ITER,
            seed: int = RNG_SEED,
            cost_name: str = COST_NAME,
            weight: float | None = None,
            verbose: bool = True) -> dict:
    """Klasyczne PSO z inertia weight. Zwraca dict z historia + gbest."""
    rng = np.random.default_rng(seed)

    lb = np.array([LOG_WI_BOUNDS[0], LOG_FC_BOUNDS[0]])
    ub = np.array([LOG_WI_BOUNDS[1], LOG_FC_BOUNDS[1]])
    span = ub - lb
    v_max = V_FRAC * span

    # Inicjalizacja: pozycja w (lb, ub), predkosc w (-v_max, v_max)
    X = rng.uniform(lb, ub, size=(n_particles, 2))
    V = rng.uniform(-v_max, v_max, size=(n_particles, 2))

    # Ocena startowa
    F = np.array([evaluate(x[0], x[1], cost_name, weight) for x in X])

    pbest_X = X.copy()
    pbest_F = F.copy()

    g_idx = int(np.argmin(F))
    gbest_X = X[g_idx].copy()
    gbest_F = float(F[g_idx])

    # Historia
    history_gbest_F = np.zeros(n_iter + 1)
    history_gbest_X = np.zeros((n_iter + 1, 2))
    history_X = np.zeros((n_iter + 1, n_particles, 2))   # caly roj w kazdej iteracji
    history_gbest_F[0] = gbest_F
    history_gbest_X[0] = gbest_X
    history_X[0] = X

    if verbose:
        print(f"PSO start: cost={cost_name}, N={n_particles}, iter={n_iter}")
        print(f"  iter  0  gbest_F = {gbest_F:.6g}  "
              f"(wi={10**gbest_X[0]:.3f}, fc={10**gbest_X[1]:.0f}Hz)")

    t0 = time.time()

    for it in range(1, n_iter + 1):
        # Linearnie malejaca waga bezwladnosci
        w = W_MAX - (W_MAX - W_MIN) * (it - 1) / max(1, n_iter - 1)

        # Update predkosci
        r1 = rng.random((n_particles, 2))
        r2 = rng.random((n_particles, 2))
        V = (w * V
             + C1 * r1 * (pbest_X - X)
             + C2 * r2 * (gbest_X[None, :] - X))
        # Klamruj predkosc
        V = np.clip(V, -v_max, v_max)

        # Update pozycji z odbiciem od granic
        X = X + V
        for d in range(2):
            below = X[:, d] < lb[d]
            above = X[:, d] > ub[d]
            X[below, d] = lb[d] + (lb[d] - X[below, d])
            X[above, d] = ub[d] - (X[above, d] - ub[d])
            V[below, d] *= -0.5     # odbicie z tlumieniem
            V[above, d] *= -0.5
        X = np.clip(X, lb, ub)

        # Ocena
        F = np.array([evaluate(x[0], x[1], cost_name, weight) for x in X])

        # Update pbest
        improved = F < pbest_F
        pbest_X[improved] = X[improved]
        pbest_F[improved] = F[improved]

        # Update gbest
        g_idx = int(np.argmin(pbest_F))
        if pbest_F[g_idx] < gbest_F:
            gbest_F = float(pbest_F[g_idx])
            gbest_X = pbest_X[g_idx].copy()

        history_gbest_F[it] = gbest_F
        history_gbest_X[it] = gbest_X
        history_X[it] = X

        if verbose and (it % 5 == 0 or it == n_iter):
            wi_b = 10 ** gbest_X[0]
            fc_b = 10 ** gbest_X[1]
            print(f"  iter {it:2d}  gbest_F = {gbest_F:.6g}  "
                  f"(wi={wi_b:.3f}, fc={fc_b:.0f}Hz)  w={w:.2f}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nPSO zakonczone w {elapsed:.1f}s  "
              f"({n_particles*(n_iter+1)} ewaluacji)")

    # Policz metryki dla optimum
    wi_opt = 10 ** gbest_X[0]
    fc_opt = 10 ** gbest_X[1]
    base = default_config()
    new_ctrl = replace(base.controller, wi=wi_opt, fc_lpf=fc_opt,
                        current_correction=CURRENT_CORRECTION)
    cfg = replace(base, controller=new_ctrl, scenario=SCENARIO, T_end=T_END)
    res = Simulator(cfg).run()
    u_ref_final = (SCENARIO.ref_step_value
                   if (SCENARIO is not None and SCENARIO.ref_step_value is not None)
                   else cfg.controller.u_ref)
    # Settling/M_p/e_ss mierzone na segmencie po ostatnim evencie (jak w gridzie)
    if SCENARIO is not None:
        ev = [tev for tev in (SCENARIO.load_step_time, SCENARIO.ref_step_time)
              if tev is not None]
        t0_m = max(ev) if ev else 0.0
    else:
        t0_m = 0.0
    seg = res["t"] >= t0_m
    t_seg = res["t"][seg] - t0_m
    final_metrics = metrics.summary(t_seg, res["v_C"][seg], res["i_L"][seg],
                                     res["s"][seg], u_ref_final)

    return {
        "cost_name": cost_name,
        "gbest_F": gbest_F,
        "wi_opt": wi_opt,
        "fc_opt": fc_opt,
        "metrics": final_metrics,
        "history_gbest_F": history_gbest_F,
        "history_gbest_X": history_gbest_X,
        "history_X": history_X,
        "elapsed_s": elapsed,
        "n_evals": n_particles * (n_iter + 1),
    }


def main() -> None:
    result = run_pso()
    # Zapis wynikow
    np.savez("optymalizacja/pso_results.npz",
             history_gbest_F=result["history_gbest_F"],
             history_gbest_X=result["history_gbest_X"],
             history_X=result["history_X"],
             gbest_F=result["gbest_F"],
             wi_opt=result["wi_opt"],
             fc_opt=result["fc_opt"],
             cost_name=result["cost_name"])
    print(f"\nZapisano: optymalizacja/pso_results.npz")
    print(f"\n{'='*70}")
    print(f"WYNIK PSO ({result['cost_name']})")
    print(f"{'='*70}")
    print(f"  wi*       = {result['wi_opt']:.4f}")
    print(f"  fc*       = {result['fc_opt']:.1f} Hz")
    print(f"  J*        = {result['gbest_F']:.6g}")
    m = result["metrics"]
    print(f"  M_p       = {m['M_p_pct']:.3f} %")
    print(f"  t_s       = {m['t_s_ms']:.3f} ms")
    print(f"  e_ss      = {m['e_ss_V']:.4f} V")
    print(f"  ripple    = {m['ripple_V']:.3f} V")
    print(f"  iL_peak   = {m['iL_peak_A']:.2f} A")
    print(f"  n_switch  = {m['n_switch']}")
    print(f"  ewaluacje = {result['n_evals']}  (czas {result['elapsed_s']:.1f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
