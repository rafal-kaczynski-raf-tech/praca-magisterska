"""Kandydaci na funkcje celu dla optymalizatora PSO.

Wszystkie funkcje przyjmuja te same argumenty (t, u_dc, i_L, s, u_ref)
i zwracaja jedna liczbe -- im mniejsza, tym lepsze nastawy.

UWAGA: u_ref moze byc skalarem LUB np.ndarray o ksztalcie t (gdy scenariusz
zawiera skok referencji). Wszystkie funkcje broadcastuja prawidlowo.

Wzory uzasadnione w literaturze klasycznej regulacji:
    - MSE/ISE: kara kwadratowa za blad sterowania (proste, ale "uśrednia szczyty")
    - IAE: kara liniowa (bardziej rownomierna waga bledow)
    - ITAE: kara liniowa wazona czasem (klasyk ze szkoly Grahama, kara dlugich oscylacji)
    - Asymetryczna: oversoot bardziej szkodliwy niz undershoot w boost converterze
    - Composite: bezposrednio celuje w klasyczne metryki (M_p, t_s, e_ss)

References:
    - Graham, D. & Lathrop, R. "The Synthesis of Optimum Transient Response",
      AIEE Trans., 1953 (kryteria IAE/ITAE/ISE/ITSE).
    - Ogata, K. "Modern Control Engineering", 5th ed., 2010.
"""
from __future__ import annotations
import numpy as np
from . import metrics


def _as_array(u_ref, t):
    """Pomocnik: broadcast u_ref do tablicy o ksztalcie t."""
    if np.isscalar(u_ref):
        return np.full_like(t, float(u_ref))
    return np.asarray(u_ref, dtype=float)


def mse(t, u_dc, i_L, s, u_ref) -> float:
    """Sredni blad kwadratowy MSE = 1/N * sum((u - u_ref)^2)."""
    u_ref_arr = _as_array(u_ref, t)
    return float(np.mean((u_dc - u_ref_arr) ** 2))


def iae(t, u_dc, i_L, s, u_ref) -> float:
    """Integral of Absolute Error: IAE = integral |e| dt."""
    u_ref_arr = _as_array(u_ref, t)
    return float(np.trapezoid(np.abs(u_dc - u_ref_arr), t))


def itae(t, u_dc, i_L, s, u_ref) -> float:
    """Integral of Time-weighted Absolute Error: ITAE = integral t*|e| dt."""
    u_ref_arr = _as_array(u_ref, t)
    return float(np.trapezoid(t * np.abs(u_dc - u_ref_arr), t))


def asymmetric(t, u_dc, i_L, s, u_ref, overshoot_penalty: float = 5.0) -> float:
    """Asymetryczna: overshoot karany overshoot_penalty x bardziej niz undershoot."""
    u_ref_arr = _as_array(u_ref, t)
    e = u_dc - u_ref_arr
    pos = np.maximum(e, 0.0)         # overshoot
    neg = np.maximum(-e, 0.0)        # undershoot
    return float(np.trapezoid(overshoot_penalty * pos ** 2 + neg ** 2, t))


def composite(t, u_dc, i_L, s, u_ref,
              alpha_ts: float = 100.0,
              beta_mp: float = 10.0,
              gamma_ess: float = 50.0) -> float:
    """Wielokryterialna: J = alpha*t_s + beta*M_p + gamma*e_ss.

    Wagi dobrane tak, by kazda skladowa miala podobny rzad wielkosci (~1-10).
    Dla scenariusza z eventami metryki sa liczone na CALYM przebiegu
    wzgledem KONCOWEGO setpointu (skalarnie - to co regulator powinien osiagnac).
    """
    u_ref_arr = _as_array(u_ref, t)
    u_ref_final = float(u_ref_arr[-1])
    t_s = metrics.settling_time(t, u_dc, u_ref_final) * 1e3   # ms
    M_p = metrics.overshoot(u_dc, u_ref_final)                # %
    e_ss = metrics.steady_state_error(u_dc, u_ref_final)      # V
    return alpha_ts * t_s + beta_mp * M_p + gamma_ess * e_ss


# Rejestr funkcji celu -- klucz uzywany w raportach i kolumnach tabel
COST_FUNCTIONS = {
    "MSE": mse,
    "IAE": iae,
    "ITAE": itae,
    "Asymmetric": asymmetric,
    "Composite": composite,
}
