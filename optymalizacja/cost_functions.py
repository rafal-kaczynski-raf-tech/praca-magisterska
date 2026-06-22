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


# Waga czlonu pradowego w funkcji celu swiadomej pradu (wariant 1 prof. Iwanskiego).
# Dobrana tak, by IAE napiecia i IAE pradu mialy podobny rzad wielkosci.
# Strojona recznie - to wybor "fizyczny" (jak wazny jest prad wzgledem napiecia).
LAMBDA_I = 1.0


def current_aware(t, u_dc, i_L, s, u_ref, i_des=None, lam: float = LAMBDA_I) -> float:
    """Wariant 1 (wg uwag prof. Iwanskiego): laczy uchyb napiecia i pradu.

        J = IAE_u + lam * IAE_i
        IAE_u = integral |u_ref - u_dc| dt          (uchyb napiecia)
        IAE_i = integral |i_des - i_L| dt           (uchyb sledzenia pradu)

    Uchyb pradu liczony jako roznica miedzy pradem pozadanym cewki (i_des,
    z bilansu mocy w sterowniku) a pradem mierzonym (i_L).

    UWAGA (pulapka stanu przejsciowego): podczas skoku referencji uklad
    CELOWO podaza za pradem maksymalnym, nie za i_des -> uchyb pradu jest tam
    duzy z definicji. Ten wariant kara go mimo to (pelny horyzont). Sluzy
    wlasnie do zaobserwowania tego efektu, zgodnie z sugestia prof. Iwanskiego;
    wariant 2 (penalizacja krotkoterminowych oscylacji) omija ta pulapke.
    """
    u_ref_arr = _as_array(u_ref, t)
    j_u = float(np.trapezoid(np.abs(u_dc - u_ref_arr), t))   # IAE napiecia
    if i_des is None:
        return j_u
    e_i = np.asarray(i_des, dtype=float) - i_L
    j_i = float(np.trapezoid(np.abs(e_i), t))                # IAE pradu
    return j_u + lam * j_i


# Waga czlonu oscylacyjnego (wariant 2 prof. Iwanskiego).
# Strojona recznie, by IAE napiecia i kara oscylacji mialy podobny rzad wielkosci.
MU_OSC = 0.02
OSC_WINDOW = 4   # liczba kolejnych probek w oknie (sugestia prof.: ~4)


def current_oscillation(t, u_dc, i_L, s, u_ref, iL_ctrl=None,
                        mu: float = MU_OSC, window: int = OSC_WINDOW) -> float:
    """Wariant 2 (wg uwag prof. Iwanskiego): kara za krotkoterminowe oscylacje pradu.

        J = IAE_u + mu * srednia_po_oknach( (max - min w oknie 4 probek)^2 )

    Czlon oscylacyjny to ROZSTEP (max-min) pradu w przesuwnym oknie 4 kolejnych
    probek w takcie sterownika (iL_ctrl) - dokladnie "roznice z czterech
    kolejnych probek", o ktorych pisal prof. Iwanski. Mierzy lokalne wahanie
    pradu (amplitude drzenia z probki na probke).

    Dlaczego okno, a nie pelny horyzont (jak wariant 1)? Bo miara lokalna jest
    prawie nieczula na powolny, celowy narost pradu podczas skoku referencji
    (w oknie 40 us prad zmienia sie nieznacznie), a silnie reaguje na szybkie
    oscylacje. Dzieki temu wariant 2 omija pulapke, na ktora wskazal profesor:
    nie karze reakcji na pradzie maksymalnym, tylko niepotrzebne oscylacje.

    Uwaga (do uzgodnienia z prof.): empirycznie sama 1./2. roznica probek nie
    rozroznia rozwiazan (aliasing ripple przy takcie 10 us); dopiero rozstep w
    oknie 4 probek poprawnie wskazuje rozwiazania o duzej amplitudzie wahan.
    """
    u_ref_arr = _as_array(u_ref, t)
    j_u = float(np.trapezoid(np.abs(u_dc - u_ref_arr), t))   # IAE napiecia
    if iL_ctrl is None:
        return j_u
    c = np.asarray(iL_ctrl, dtype=float)
    w = int(window)
    if c.size < w:
        return j_u
    # przesuwne okno dlugosci w: rozstep (max - min) w kazdym oknie
    win = np.lib.stride_tricks.sliding_window_view(c, w)      # (N-w+1, w)
    spread = win.max(axis=1) - win.min(axis=1)               # lokalny rozstep
    j_osc = float(np.mean(spread ** 2))                       # srednia kwadratu
    return j_u + mu * j_osc


# Rejestr funkcji celu -- klucz uzywany w raportach i kolumnach tabel
COST_FUNCTIONS = {
    "MSE": mse,
    "IAE": iae,
    "ITAE": itae,
    "Asymmetric": asymmetric,
    "Composite": composite,
    "CurrentAware": current_aware,
    "CurrentOscillation": current_oscillation,
}

# Funkcje celu wymagajace i_des (uchyb sledzenia pradu) - obsluga w pso.evaluate
CURRENT_AWARE = {"CurrentAware"}

# Funkcje celu wymagajace pradu w takcie sterownika (iL_ctrl) - druga roznica
OSCILLATION_AWARE = {"CurrentOscillation"}
