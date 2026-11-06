"""ZOH sampler - modeluje opoznienie 1 probki na pomiarach sterownika.

Sterownik widzi i_L(t-Ts), v_C(t-Ts) zamiast aktualnych wartosci.
Klasa hermetyzuje te dwie wartosci buforowane.
"""
from __future__ import annotations


class ZOHSampler:
    """1-sample delay dla pomiarow (i_L, v_C)."""

    def __init__(self, i_L0: float, v_C0: float) -> None:
        self.i_L_prev: float = i_L0
        self.v_C_prev: float = v_C0

    def read(self) -> tuple[float, float]:
        """Zwraca pomiary widoczne przez sterownik w tej chwili (z opoznieniem)."""
        return self.i_L_prev, self.v_C_prev

    def latch(self, i_L_now: float, v_C_now: float) -> None:
        """Zatrzasniecie aktualnych wartosci na nastepny tick sterownika."""
        self.i_L_prev = i_L_now
        self.v_C_prev = v_C_now
