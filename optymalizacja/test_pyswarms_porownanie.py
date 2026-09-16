"""Test porownawczy: Wlasne PSO vs PySwarms dla optymalizacji dwuetapowej 2D.

Ten skrypt nie modyfikuje zadnych istniejacych plikow ani wynikow.
Sluzy do precyzyjnej weryfikacji, na ile optymalizator z biblioteki PySwarms
(GlobalBestPSO) zbiega sie do tych samych nastaw co nasza implementacja referencyjna.
"""
from __future__ import annotations
import numpy as np
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO

from optymalizacja.optymalizacja_dwuetapowa_2d import (
    evaluate_stage1, evaluate_stage2,
    run_stage1_pso, run_stage2_pso,
    N_PARTICLES, N_ITER, W_MAX, W_MIN, C1, C2, SEED
)

def run_pyswarms_stage1() -> tuple[float, float, float]:
    print("\n=======================================================")
    print(">>> Uruchomienie KROKU 1 przez PySwarms (GlobalBestPSO)")
    print("=======================================================")
    # Granice logarytmiczne: wi in [0.05, 2.0], fc_i in [100, 2000] Hz
    lb = np.array([np.log10(0.05), np.log10(100.0)])
    ub = np.array([np.log10(2.0),  np.log10(2000.0)])
    bounds = (lb, ub)

    # Funkcja celu przyjmujaca macierz czastek (N, 2)
    def batch_eval(X: np.ndarray) -> np.ndarray:
        return np.array([
            evaluate_stage1(float(10.0 ** x[0]), float(10.0 ** x[1]))
            for x in X
        ])

    options = {"c1": C1, "c2": C2, "w": W_MAX}
    oh_strategy = {"w": "lin_variation"}
    
    # Inicjalizacja punktu bazowego
    init_pos = np.random.default_rng(SEED).uniform(lb, ub, size=(N_PARTICLES, 2))
    init_pos[0] = np.array([np.log10(0.50), np.log10(321.0)])

    optimizer = GlobalBestPSO(
        n_particles=N_PARTICLES,
        dimensions=2,
        options=options,
        bounds=bounds,
        bh_strategy="reflective",
        oh_strategy=oh_strategy,
        init_pos=init_pos,
        ftol=-np.inf,
    )

    cost, pos = optimizer.optimize(batch_eval, iters=N_ITER, verbose=True)
    best_wi = float(10.0 ** pos[0])
    best_fci = float(10.0 ** pos[1])
    print(f"--> PySwarms KROK 1: wi* = {best_wi:.4f}, fc_i* = {best_fci:.1f} Hz (J1 = {cost:.4f})")
    return best_wi, best_fci, cost

def run_pyswarms_stage2(wi_fixed: float, fci_fixed: float) -> tuple[float, float, float]:
    print("\n=======================================================")
    print(">>> Uruchomienie KROKU 2 przez PySwarms (GlobalBestPSO)")
    print("=======================================================")
    # Granice logarytmiczne: wu in [1.0, 35.0], fc_u in [100, 2000] Hz
    lb = np.array([np.log10(1.0),  np.log10(100.0)])
    ub = np.array([np.log10(35.0), np.log10(2000.0)])
    bounds = (lb, ub)

    def batch_eval(X: np.ndarray) -> np.ndarray:
        return np.array([
            evaluate_stage2(float(10.0 ** x[0]), float(10.0 ** x[1]), wi_fixed, fci_fixed)[1]
            for x in X
        ])

    options = {"c1": C1, "c2": C2, "w": W_MAX}
    oh_strategy = {"w": "lin_variation"}

    init_pos = np.random.default_rng(SEED + 1).uniform(lb, ub, size=(N_PARTICLES, 2))
    init_pos[0] = np.array([np.log10(20.0), np.log10(321.0)])

    optimizer = GlobalBestPSO(
        n_particles=N_PARTICLES,
        dimensions=2,
        options=options,
        bounds=bounds,
        bh_strategy="reflective",
        oh_strategy=oh_strategy,
        init_pos=init_pos,
        ftol=-np.inf,
    )

    cost, pos = optimizer.optimize(batch_eval, iters=N_ITER, verbose=True)
    best_wu = float(10.0 ** pos[0])
    best_fcu = float(10.0 ** pos[1])
    print(f"--> PySwarms KROK 2: wu* = {best_wu:.4f}, fc_u* = {best_fcu:.1f} Hz (J2 = {cost:.4f})")
    return best_wu, best_fcu, cost

def main():
    print("############################################################")
    print("POROWNANIE: Wlasne PSO (Reference) vs PySwarms (GlobalBestPSO)")
    print("############################################################\n")

    # 1. KROK 1: Nasze PSO
    print(">>> 1. KROK 1: Wlasna implementacja PSO")
    ref_wi, ref_fci, ref_j1 = run_stage1_pso()

    # 2. KROK 1: PySwarms
    ps_wi, ps_fci, ps_j1 = run_pyswarms_stage1()

    print("\n------------------------------------------------------------")
    print(f"WYNIKI KROK 1:")
    print(f"  Referencja (Custom PSO): wi* = {ref_wi:.4f}, fc_i* = {ref_fci:.1f} Hz, J1 = {ref_j1:.5f}")
    print(f"  Biblioteka (PySwarms):  wi* = {ps_wi:.4f}, fc_i* = {ps_fci:.1f} Hz, J1 = {ps_j1:.5f}")
    print(f"  Delta J1:               {abs(ref_j1 - ps_j1):.6f} ({(abs(ref_j1 - ps_j1)/ref_j1)*100:.2f}%)")
    print("------------------------------------------------------------\n")

    # 3. KROK 2: Wlasne PSO (z zamrozonymi nastawami z referencji)
    print(">>> 2. KROK 2: Wlasna implementacja PSO")
    ref_wu, ref_fcu, ref_j2 = run_stage2_pso(ref_wi, ref_fci)

    # 4. KROK 2: PySwarms (z zamrozonymi nastawami z KROKU 1 PySwarms)
    ps_wu, ps_fcu, ps_j2 = run_pyswarms_stage2(ps_wi, ps_fci)

    print("\n------------------------------------------------------------")
    print(f"WYNIKI KROK 2:")
    print(f"  Referencja (Custom PSO): wu* = {ref_wu:.4f}, fc_u* = {ref_fcu:.1f} Hz, J2 = {ref_j2:.5f}")
    print(f"  Biblioteka (PySwarms):  wu* = {ps_wu:.4f}, fc_u* = {ps_fcu:.1f} Hz, J2 = {ps_j2:.5f}")
    print(f"  Delta J2:               {abs(ref_j2 - ps_j2):.6f} ({(abs(ref_j2 - ps_j2)/ref_j2)*100:.2f}%)")
    print("------------------------------------------------------------")

if __name__ == "__main__":
    main()
