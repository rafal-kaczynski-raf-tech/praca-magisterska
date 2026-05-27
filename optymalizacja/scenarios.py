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
