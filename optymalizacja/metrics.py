"""Klasyczne metryki jakosci regulacji.

Obliczane na podstawie przebiegu napiecia wyjsciowego u_dc(t) wzgledem
wartosci zadanej u_ref. Wszystkie metryki sa "single-shot" -- jedna liczba
na jedna symulacje.

References:
    Ogata, K. "Modern Control Engineering", 5th ed., 2010, ch. 5.
"""
from __future__ import annotations
import numpy as np


def overshoot(u: np.ndarray, u_ref: float) -> float:
    """Przeregulowanie M_p [%] = max((u - u_ref) / u_ref) * 100.

    Zwraca 0 jesli przebieg nigdy nie przekroczyl wartosci zadanej.
    """
    peak = float(np.max(u))
    return max(0.0, (peak - u_ref) / u_ref * 100.0)


def settling_time(t: np.ndarray, u: np.ndarray, u_ref: float,
                  tol_pct: float = 2.0) -> float:
    """Czas ustalania t_s -- moment, po ktorym |u - u_ref| < tol_pct * u_ref.

    Zwraca t[-1] (caly horyzont) jesli nigdy nie wszedl do strefy tolerancji.
    """
    tol_abs = tol_pct / 100.0 * u_ref
    inside = np.abs(u - u_ref) < tol_abs
    # ostatni indeks, dla ktorego TROCHE PO NIM jeszcze sie zdarzylo wyjsc poza strefe
    outside_idx = np.where(~inside)[0]
    if len(outside_idx) == 0:
        return float(t[0])
    last_outside = outside_idx[-1]
    if last_outside >= len(t) - 1:
        return float(t[-1])
    return float(t[last_outside + 1])


def steady_state_error(u: np.ndarray, u_ref: float, last_frac: float = 0.2) -> float:
    """Blad ustalony e_ss = |mean(u_last_window) - u_ref|.

    Args:
        last_frac: ulamek konca przebiegu liczony jako "stan ustalony" (default 20%).
    """
    n = len(u)
    i0 = int(n * (1.0 - last_frac))
    return float(abs(np.mean(u[i0:]) - u_ref))


def ripple_pp(u: np.ndarray, last_frac: float = 0.2) -> float:
    """Tetnienia peak-peak w stanie ustalonym (ostatnie last_frac % przebiegu)."""
    n = len(u)
    i0 = int(n * (1.0 - last_frac))
    window = u[i0:]
    return float(np.max(window) - np.min(window))


def peak_current(i_L: np.ndarray) -> float:
    """Wartosc szczytowa pradu dlawika -- ogranicznik bezpieczenstwa."""
    return float(np.max(np.abs(i_L)))


def switching_count(s: np.ndarray) -> int:
    """Liczba zdarzen przelaczenia tranzystora -- proxy strat komutacji."""
    return int(np.sum(np.abs(np.diff(s.astype(np.int32)))))


def summary(t: np.ndarray, u_dc: np.ndarray, i_L: np.ndarray, s: np.ndarray,
            u_ref: float) -> dict:
    """Komplet metryk dla jednej symulacji."""
    return {
        "M_p_pct": overshoot(u_dc, u_ref),
        "t_s_ms": settling_time(t, u_dc, u_ref) * 1e3,
        "e_ss_V": steady_state_error(u_dc, u_ref),
        "ripple_V": ripple_pp(u_dc),
        "iL_peak_A": peak_current(i_L),
        "n_switch": switching_count(s),
    }
