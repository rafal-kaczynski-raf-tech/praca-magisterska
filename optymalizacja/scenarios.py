"""Scenariusze testowe do badania funkcji celu.

Default scenario (None) = monotoniczny rozruch 100 -> 240 V, R_load = 47.6 Ohm
przez caly horyzont T_end = 100 ms. Krajobraz funkcji celu okazal sie bardzo
plaski - wszystkie kandydaty wskazywaly praktycznie identyczne nastawy.

hard_scenario() = wzbogacenie scenariusza o:
  - skok ODCIAZENIA (R_load 47.6 -> 95.2 Ohm) w t = 50 ms - obciazenie x0.5,
    napiecie tendencyjnie rosnie -> regulator musi pchnac s=0 i zduszic je
  - skok wartosci zadanej w GORE (u_ref 240 -> 260 V) w t = 100 ms - wymaga
    pchania pradu, ale bez naruszenia OCP (przy R=95.2 Ohm: P=260^2/95.2=710W,
    i_in=7.1A << i_max=20A)
  - dluzszy horyzont T_end = 200 ms (50 ms na rozruch + 50 ms steady + 50 ms
    odpowiedzi na load step + 100 ms odpowiedzi na ref step)

UWAGA: probowano wczesniej wariantu z dociazaniem (R/2) + ref step W DOL.
Okazalo sie, ze nastawy regulatora nie maja wplywu, bo dociaznie aktywuje
OCP (i_max=20A), a skok w dol pasywnie rozladowuje kondensator. Aktualne
wybory wymuszaja AKTYWNE przelaczanie zalezne od (wi, fc_lpf).
"""
from __future__ import annotations
from src.config import Scenario


HARD_T_END = 0.20  # s


def default_scenario() -> None:
    """Brak zdarzen - sterownik widzi tylko jeden skok startowy 100 -> 240 V."""
    return None


def hard_scenario() -> Scenario:
    """Trudny scenariusz: load off (R x2) + reference step w gore (modest)."""
    return Scenario(
        load_step_time=50e-3,
        load_step_R=95.238,         # R_load x2 -> obciazenie x0.5 (load off)
        ref_step_time=100e-3,
        ref_step_value=260.0,       # +20 V w gore z 240 V
    )


STRESS_T_END = 0.20  # s

# Ciag skokow referencji co 20 ms: 240 <-> 280 V (8 przejsc).
# Kazdy skok W GORE (240->280) wymusza surge pradu (ladowanie C), kazdy skok
# W DOL (280->240) wymaga zduszenia. Przy R=47.6 Ohm i 280 V: P=1647 W,
# i_in=16.5 A < i_max=20 A -> brak nasycenia OCP, nastawy (wi, fc) maja wplyw.
# Cel: transienty pradowe zajmuja DUZA czesc horyzontu (w przeciwienstwie do
# hard_scenario, gdzie pojedynczy transient to ~5% calki bledu pradu) -> tu
# "pulapka" prof. (kara za blad pradu) FAKTYCZNIE zaczyna spowalniac sterownik.
_STRESS_PULSES = (
    (0.040, 280.0),
    (0.060, 240.0),
    (0.080, 280.0),
    (0.100, 240.0),
    (0.120, 280.0),
    (0.140, 240.0),
    (0.160, 280.0),
    (0.180, 240.0),
)


def stress_scenario() -> Scenario:
    """Stress-test: powtarzalne skoki referencji 240<->280 V co 20 ms.

    Wymusza serie surge'ow pradu rozsianych po calym horyzoncie. W tym
    scenariuszu transienty pradowe stanowia duza czesc calego bledu pradu,
    wiec funkcja celu swiadoma pradu (CurrentAware) zaczyna preferowac
    wolniejsza, lagodniejsza odpowiedz - "pulapka" sie budzi.
    """
    return Scenario(ref_pulses=_STRESS_PULSES)


PROF_T_END = 0.30  # s

R_LOAD_NOMINAL = 47.619   # Ohm - nominalne obciazenie ("urzadzenie" podlaczone)
R_LOAD_OFF = 1.0e7        # Ohm - "urzadzenie odlaczone" (praktyczny rozwarcie)

# Wg wytycznych prof. Iwanskiego (spotkanie Teams): odlaczenie/podlaczenie
# obciazenia (R_load) symulujace wylaczenie/wlaczenie "urzadzenia".
_PROF_LOAD_PULSES = (
    (0.05, R_LOAD_OFF),        # t=50ms:  wylaczenie urzadzenia
    (0.10, R_LOAD_NOMINAL),    # t=100ms: wlaczenie urzadzenia
    (0.20, R_LOAD_OFF),        # t=200ms: wylaczenie urzadzenia
    (0.25, R_LOAD_NOMINAL),    # t=250ms: wlaczenie urzadzenia
)


def professor_scenario() -> Scenario:
    """Scenariusz wg wytycznych prof. Iwanskiego (spotkanie Teams, lipiec 2026).

    Oś czasu (v_C0 = 100 V, rozruch do u_ref = 240 V jak dotychczas):
      t=0.00s  start, rozruch 100 -> 240 V
      t=0.05s  wylaczenie urzadzenia (odlaczenie R_load)
      t=0.10s  wlaczenie urzadzenia (powrot R_load)
      t=0.15s  skok referencji 240 V -> 160 V
      t=0.20s  wylaczenie urzadzenia
      t=0.25s  wlaczenie urzadzenia
      t=0.30s  koniec pomiaru / probkowania

    Wskazniki interesujace prof. liczy sie WYLACZNIE w stanie ustalonym
    (poza oknem przejsciowym po kazdym zdarzeniu):
      1) uchyb napiecia wzgledem wartosci zadanej (mean(v_C) - u_ref),
      2) tetnienia pradu wzgledem wartosci sredniej (ripple_pp / mean(i_L) * 100%).

    Patrz optymalizacja/analiza_scenariusz_prof.py.
    """
    return Scenario(
        ref_step_time=0.15,
        ref_step_value=160.0,
        load_pulses=_PROF_LOAD_PULSES,
    )


# ---------------------------------------------------------------------------
# Replika scenariusza Fig. 8 z artykulu (Tatari/Bizhani/Iwanski, IEEE JESTIE,
# sekcja VI-A-1 "Proposed BB Control Performance"). Parametry wg Table I.
# ---------------------------------------------------------------------------

ARTICLE_T_END = 0.45  # s - 150ms po ostatnim zdarzeniu (t=0.30s) na ustalenie

# Table I: Input Voltage vs=100V, Resistive Load Range 41.6-416 Ohm,
# Output Voltage Range vout*=150-250V, Maximum Inductor Current imax_L=20A,
# Input Inductance L=750uH, Output Capacitance C=1500uF, Sampling Time Ts=20us.
ARTICLE_V_IN = 100.0
ARTICLE_L = 0.75e-3
ARTICLE_C = 1500e-6
ARTICLE_R_INITIAL = 41.6      # Ohm - CCM
ARTICLE_R_AFTER = 208.0       # Ohm - CCM->DCM po skoku obciazenia
ARTICLE_TS = 20e-6            # s - okres probkowania sterownika
ARTICLE_I_MAX = 20.0          # A
ARTICLE_FC_LPF = 200.0        # Hz - "effective compromise" wg artykulu
ARTICLE_U_REF0 = 250.0        # V
ARTICLE_U_REF1 = 150.0        # V
ARTICLE_REF_STEP_TIME = 0.15  # s
ARTICLE_LOAD_STEP_TIME = 0.30  # s

# Empiryczna regula strojenia wagi lambda z artykulu (rown. po (21)):
#   lambda(C) = 7.48 * C[uF]^-0.547
# Dla C=1500uF: lambda ~= 0.137.
ARTICLE_LAMBDA = 7.48 * (ARTICLE_C * 1e6) ** (-0.547)


def article_fig8_scenario() -> Scenario:
    """Replika scenariusza z Fig. 8 artykulu (sekcja VI-A-1).

    Oś czasu (opisana w tekście artykulu, wout* i R w Table I):
      t=0.00s  start, u_ref=250V (CCM, R=41.6 Ohm)
      t=0.15s  skok referencji 250V -> 150V (R bez zmian)
      t=0.30s  skok obciazenia R: 41.6 -> 208 Ohm (przejscie CCM->DCM)
      t=0.45s  koniec (dodatkowe 150ms na ustalenie po ostatnim zdarzeniu,
               artykul nie podaje dokladnego konca pomiaru)

    UWAGA: to NIE jest cyfrowa replika danych z wykresu artykulu (nie mamy
    dostepu do surowych danych/kodu autorow) - to nasza symulacja z tymi
    samymi parametrami fizycznymi (Table I), tym samym rownaniem sterujacym
    (Eq.14-17, current_correction=True) i ta sama regula lambda(C), majaca
    umozliwic JAKOSCIOWE porownanie zachowania (osiadanie, tetnienia,
    uchyb ustalony) z opisem w tekscie artykulu.
    """
    return Scenario(
        ref_step_time=ARTICLE_REF_STEP_TIME,
        ref_step_value=ARTICLE_U_REF1,
        load_pulses=((ARTICLE_LOAD_STEP_TIME, ARTICLE_R_AFTER),),
    )

