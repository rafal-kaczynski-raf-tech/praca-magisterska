"""
Zapisuje "zloty" snapshot wynikow Kroku 2.6 PRZED refactoringiem do OOP.

Uruchamia oryginalna proceduralna implementacje (demo_bangbang_estim.simulate)
i zapisuje surowe tablice do tests/baseline_results.npz.

Po refactoringu nowa implementacja OOP musi reprodukowac te wyniki BIT-W-BIT
(max|delta| < 1e-12). Sprawdza to tests/test_regression.py.
"""
from pathlib import Path
import numpy as np
import sys

# Import oryginalnej proceduralnej implementacji
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import demo_bangbang_estim as orig


def main() -> None:
    out_path = Path(__file__).resolve().parent / "baseline_results.npz"

    print("Uruchamiam oryginalna proceduralna symulacje...")
    res = orig.simulate()

    np.savez_compressed(
        out_path,
        t=res["t"],
        i_L=res["i_L"],
        v_C=res["v_C"],
        s=res["s"],
        t_ctrl=res["t_ctrl"],
        iout_est=res["iout_est"],
        iout_filt=res["iout_filt"],
        i_des=res["i_des"],
        error_u=res["error_u"],
        error_i=res["error_i"],
        iL_sample=res["iL_sample"],
        vC_sample=res["vC_sample"],
        # Parametry zamrozone razem z danymi (do sanity check w tescie)
        params=np.array([
            orig.V_in, orig.L, orig.R_L, orig.C, orig.R_load, orig.u_ref,
            orig.v_C0, orig.i_L0,
            orig.T_s_ctrl, orig.wi, orig.i_max,
            orig.fc_lpf, orig.fs_lpf,
            orig.dt_phys, float(orig.N_per_ctrl),
            orig.T_END, float(orig.N_steps),
            orig.B0, orig.B1, orig.B2, orig.A1, orig.A2,
        ], dtype=np.float64),
    )
    print(f"\nZapisano baseline: {out_path}")
    print(f"  Probek fizyki:    {len(res['t'])}")
    print(f"  Probek sterownika:{len(res['t_ctrl'])}")
    print(f"  iL[-1] = {res['i_L'][-1]:.15e}")
    print(f"  vC[-1] = {res['v_C'][-1]:.15e}")


if __name__ == "__main__":
    main()
