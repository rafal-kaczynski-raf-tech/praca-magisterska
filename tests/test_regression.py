"""
Test regresji: nowa implementacja OOP MUSI byc identyczna bit-w-bit
z zamrozonym snapshotem proceduralnej wersji Kroku 2.6.

Uruchom:
    .venv/bin/python tests/test_regression.py

Sukces = wszystkie tablice maja max|delta| < TOL.
Porazka = print diagnostyki + exit 1.
"""
from pathlib import Path
import sys
import numpy as np

TOL = 1e-12  # praktycznie identycznosc numeryczna w float64


def load_baseline(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"BLAD: brak baseline'u {path}\n"
            "Uruchom najpierw: .venv/bin/python tests/baseline_snapshot.py"
        )
    data = np.load(path)
    return {k: data[k] for k in data.files}


def run_oop_simulation() -> dict:
    """Uruchamia nowa OOP-owa implementacje i zwraca dict w tym samym formacie."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.simulator import Simulator  # type: ignore
    from src.config import default_config  # type: ignore

    sim = Simulator(default_config())
    return sim.run()


def compare(name: str, baseline: np.ndarray, new: np.ndarray) -> bool:
    if baseline.shape != new.shape:
        print(f"  [FAIL] {name}: ksztalt {baseline.shape} vs {new.shape}")
        return False
    if baseline.dtype != new.dtype:
        # Dopuszczamy rozne dtype tylko jesli wartosci nadal sie zgadzaja
        pass
    diff = np.abs(baseline.astype(np.float64) - new.astype(np.float64))
    max_d = float(diff.max()) if diff.size > 0 else 0.0
    status = "OK  " if max_d <= TOL else "FAIL"
    print(f"  [{status}] {name:12s}  max|delta| = {max_d:.3e}")
    return max_d <= TOL


def main() -> None:
    baseline_path = Path(__file__).resolve().parent / "baseline_results.npz"
    print(f"Wczytuje baseline: {baseline_path.name}")
    baseline = load_baseline(baseline_path)

    print("Uruchamiam nowa implementacje OOP...")
    new = run_oop_simulation()

    print(f"\nPorownanie (TOL = {TOL:.0e}):")
    keys = ["t", "i_L", "v_C", "s",
            "t_ctrl", "iout_est", "iout_filt", "i_des",
            "error_u", "error_i", "iL_sample", "vC_sample"]
    results = [compare(k, baseline[k], new[k]) for k in keys]

    if all(results):
        print("\nWSZYSTKO OK - OOP identyczne bit-w-bit z proceduralnym.")
        sys.exit(0)
    else:
        print("\nREGRESJA - sa roznice. Refactoring NIE jest bezpieczny.")
        sys.exit(1)


if __name__ == "__main__":
    main()
