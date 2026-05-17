"""Fizyka obwodu synchronicznego boost converter.

Pojedynczy krok Eulera w przod (forward Euler), dt rzedu 0.5 us.
Klasa nie wie nic o sterowniku - zna tylko biezaca decyzje klucza s in {0, 1}.

Rownania (synchronous boost, s = stan kluczy):
    v_L = V_in - (1-s) * v_C - R_L * i_L
    i_C = (1-s) * i_L - v_C / R_load
    di_L/dt = v_L / L
    dv_C/dt = i_C / C
"""
from __future__ import annotations
from .config import ConverterParams


class Converter:
    """Stan przekształtnika + krok Eulera."""

    def __init__(self, params: ConverterParams) -> None:
        self.p = params
        self.i_L: float = params.i_L0
        self.v_C: float = params.v_C0

    def step(self, s: int, dt: float) -> None:
        """Jeden krok Eulera w przod przy stanie klucza s.

        UWAGA: kolejnosc operacji 1:1 z proceduralna implementacja
        (krytyczne dla identycznosci bit-w-bit).
        """
        p = self.p
        v_L = p.V_in - (1.0 - s) * self.v_C - p.R_L * self.i_L
        i_C = (1.0 - s) * self.i_L - self.v_C / p.R_load
        self.i_L = self.i_L + v_L / p.L * dt
        self.v_C = self.v_C + i_C / p.C * dt
