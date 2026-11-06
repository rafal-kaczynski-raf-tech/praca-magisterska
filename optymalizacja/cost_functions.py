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


# Waga kary za MAGNITUDE pradu (wariant 3 - "pulapka na zywo").
# Strojona recznie, by uchwycic moment, w ktorym optymalizator zaczyna
# poswiecac regulacje napiecia, byle zmniejszyc prad.
GAMMA_EFFORT = 0.02


def current_effort(t, u_dc, i_L, s, u_ref, gamma: float = GAMMA_EFFORT) -> float:
    """Wariant 3 (kontrprzyklad): kara za MAGNITUDE pradu (wysilek sterowania).

        J = IAE_u + gamma * integral( i_L^2 ) dt

    Klasyczny czlon "control effort" (jak w regulatorze LQR) na pradzie cewki.
    W przeciwienstwie do wariantow 1 i 2 (kara za UCHYB/oscylacje, ktore daja
    sie "oszukac" agresywnym i_des) tutaj karzemy BEZWZGLEDNA wartosc pradu.

    To celowy kontrprzyklad - pokazuje "pulapke" prof. Iwanskiego NA ZYWO:
    pradu w stanie ustalonym NIE da sie obnizyc inaczej niz rezygnujac z
    dostarczania mocy (bilans: i_in = P/V_in = u_dc^2/(R*V_in)). Wraz ze
    wzrostem gamma optymalizator zaczyna wiec CELOWO niedoregulowywac napiecie
    (wieksze e_ss), byle zmniejszyc prad - dokladnie ta degeneracja, przed
    ktora ostrzegal profesor, gdy kryterium ignoruje sens fizyczny pradu.
    """
    u_ref_arr = _as_array(u_ref, t)
    j_u = float(np.trapezoid(np.abs(u_dc - u_ref_arr), t))   # IAE napiecia
    j_eff = float(np.trapezoid(np.asarray(i_L, dtype=float) ** 2, t))  # ISE pradu
    return j_u + gamma * j_eff


# Domyslne parametry dwustrefowej funkcji celu wg prof. Iwanskiego
U_THRESH_DEFAULT = 5.0    # prog napiecia [V] rozgraniczajacy stany
W_TRANS_DEFAULT = 40.0    # waga stanu przejsciowego (wyrownywujaca rzedy wielkosci 40:1)


def error_threshold_aware(t, u_dc, i_L, s, u_ref, i_des=None,
                          u_thresh: float = U_THRESH_DEFAULT,
                          i_max: float = 20.0,
                          w_trans: float = W_TRANS_DEFAULT,
                          w_steady: float = 1.0,
                          lam: float = LAMBDA_I) -> float:
    """Dwustrefowa funkcja celu zalezna od bledu napiecia (koncepcja prof. Iwanskiego).

    Dla |e_u| > u_thresh (stan przejsciowy):
        Cel to pelne nasycenie pradu cewki (+i_max dla e_u>0, -i_max dla e_u<0).
        Funkcja celu nie sprawdza bledu napiecia, tylko precyzje osiagniecia nasycenia:
        J_trans = integral( |i_target - i_L| ) dt

    Dla |e_u| <= u_thresh (otoczenie stanu ustalonego):
        Standardowe kryterium CurrentAware:
        J_steady = IAE_u + lam * IAE_i

    Koszt calkowity:
        J = w_trans * J_trans + w_steady * J_steady
    """
    u_ref_arr = _as_array(u_ref, t)
    e_u = u_ref_arr - u_dc

    large_mask = np.abs(e_u) > u_thresh
    small_mask = ~large_mask

    # 1. Strefa duzego bledu (> u_thresh): prąd w nasyceniu
    i_target = np.where(e_u > 0, i_max, -i_max)
    j_trans = float(np.trapezoid(np.where(large_mask, np.abs(i_target - i_L), 0.0), t))

    # 2. Strefa malego bledu (<= u_thresh): napiecie + prad (CurrentAware)
    j_steady_u = float(np.trapezoid(np.where(small_mask, np.abs(e_u), 0.0), t))
    if i_des is not None:
        e_i = np.asarray(i_des, dtype=float) - i_L
        j_steady_i = float(np.trapezoid(np.where(small_mask, np.abs(e_i), 0.0), t))
    else:
        j_steady_i = 0.0

    j_steady = j_steady_u + lam * j_steady_i

    return w_trans * j_trans + w_steady * j_steady


W_PEAK_DEFAULT = 0.05     # waga kary za piki przeregulowan / zapadow w transiencie


def transient_damping_aware(t, u_dc, i_L, s, u_ref, i_des=None,
                            u_thresh: float = U_THRESH_DEFAULT,
                            w_peak: float = W_PEAK_DEFAULT,
                            lam: float = LAMBDA_I,
                            stage: str = "both") -> float:
    """Ulepszona dwustrefowa funkcja celu z penalizacja zapadu/przeregulowania.

    Rozwiazuje problem przedwczesnego hamowania w transiencie:
    1. Dla |e_u| > u_thresh (w trakcie skoku):
       - liczy czas trwania skoku j_trans_time = integral(|e_u|) dt w strefie > u_thresh,
       - karze kwadratowo maksymalny zapad po skoku 240->160V oraz przeregulowanie po zalaczeniu R.
       Nie wymusza na sile ciaglego nasycenia pradem - pozwala sterownikowi wyhamowac
       prad przed celem bez nakladania sztucznej kary.
    2. Dla |e_u| <= u_thresh (stan ustalony):
       - kryterium CurrentAware: j_steady = IAE_u + lam * IAE_i.

    Parametr stage umozliwia scisla optymalizacje dwutorowa:
       - "both" (domyslnie): zwraca pelny koszt j_steady + j_trans_time + w_peak * j_peak
       - "steady": zwraca wylacznie czlon stanu ustalonego j_steady (Krok 1 w optymalizacji dwuetapowej)
       - "transient": zwraca wylacznie czlon transjentu j_trans_time + w_peak * j_peak (Krok 2)
    """
    u_ref_arr = _as_array(u_ref, t)
    e_u = u_dc - u_ref_arr

    large_mask = np.abs(e_u) > u_thresh
    small_mask = ~large_mask

    # 1. Strefa skoku (> u_thresh): calka bledu napiecia + kwadrat zapadu/przeregulowania/odbicia
    j_trans_time = float(np.trapezoid(np.where(large_mask, np.abs(e_u), 0.0), t))

    # Maksymalny zapad po skoku 240->160V w t=150ms (z tolerancja na naturalne ugiecie 0.2V):
    m150 = (t >= 0.150) & (t <= 0.155)
    raw_undershoot = max(0.0, 160.0 - float(np.min(u_dc[m150]))) if m150.any() else 0.0
    undershoot = max(0.0, raw_undershoot - 0.2)

    # Odbicie powyzej 160V po dojechaniu do celu (bada zachowanie PO dojsciu do 160V):
    m150_reb = (t >= 0.1515) & (t <= 0.155)
    rebound = max(0.0, float(np.max(u_dc[m150_reb])) - 160.0) if m150_reb.any() else 0.0

    # Maksymalne przeregulowanie po zalaczeniu obciazenia w t=100ms:
    m100 = (t >= 0.100) & (t <= 0.108)
    overshoot = max(0.0, float(np.max(u_dc[m100])) - 240.0) if m100.any() else 0.0

    j_peak = undershoot ** 2 + rebound ** 2 + overshoot ** 2
    j_trans = j_trans_time + w_peak * j_peak

    # 2. Strefa stanu ustalonego (<= u_thresh):
    j_steady_u = float(np.trapezoid(np.where(small_mask, np.abs(e_u), 0.0), t))
    if i_des is not None:
        e_i = np.asarray(i_des, dtype=float) - i_L
        j_steady_i = float(np.trapezoid(np.where(small_mask, np.abs(e_i), 0.0), t))
    else:
        j_steady_i = 0.0

    j_steady = j_steady_u + lam * j_steady_i

    if stage == "steady":
        return j_steady
    if stage == "transient":
        return j_trans
    return j_steady + j_trans


# Rejestr funkcji celu -- klucz uzywany w raportach i kolumnach tabel
COST_FUNCTIONS = {
    "MSE": mse,
    "IAE": iae,
    "ITAE": itae,
    "Asymmetric": asymmetric,
    "Composite": composite,
    "CurrentAware": current_aware,
    "CurrentOscillation": current_oscillation,
    "CurrentEffort": current_effort,
    "ErrorThresholdAware": error_threshold_aware,
    "TransientDampingAware": transient_damping_aware,
}

# Funkcje celu wymagajace i_des (uchyb sledzenia pradu) - obsluga w pso.evaluate
CURRENT_AWARE = {"CurrentAware", "ErrorThresholdAware", "TransientDampingAware"}

# Funkcje celu wymagajace pradu w takcie sterownika (iL_ctrl) - druga roznica
OSCILLATION_AWARE = {"CurrentOscillation"}

# Funkcje celu karzace NADMIAR pradu (wymagaja i_des) - obsluga w pso.evaluate
CURRENT_EFFORT = {"CurrentEffort"}
