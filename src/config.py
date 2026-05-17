"""Konfiguracja symulacji - jeden punkt zmiany parametrow.

Dataclass zamrazana (frozen=True) zeby przypadkiem nie zmienic parametrow
w trakcie symulacji.  Domyslne wartosci = stan PSIM dla Kroku 2.6.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ConverterParams:
    """Parametry fizyczne obwodu boost."""
    V_in: float = 100.0        # napiecie zrodla [V]
    L: float = 0.75e-3         # indukcyjnosc cewki [H]
    R_L: float = 0.05          # ESR cewki [Ohm]
    C: float = 200e-6          # pojemnosc kondensatora [F]
    R_load: float = 47.619     # rezystancja obciazenia (R32||R33) [Ohm]
    v_C0: float = 100.0        # warunek poczatkowy U_C [V]
    i_L0: float = 0.0          # warunek poczatkowy I_L [A]


@dataclass(frozen=True)
class ControllerParams:
    """Parametry sterownika MF-BB."""
    u_ref: float = 240.0       # napiecie zadane [V]
    T_s_ctrl: float = 10e-6    # okres probkowania sterownika [s]
    wi: float = 1.0            # waga bledu pradu
    i_max: float = 20.0        # zabezpieczenie nadpradowe [A]
    fc_lpf: float = 2000.0     # czestotliwosc odciecia LPF [Hz]


@dataclass(frozen=True)
class SimulationConfig:
    """Pelna konfiguracja symulacji."""
    converter: ConverterParams
    controller: ControllerParams
    dt_phys: float = 0.5e-6    # krok fizyki [s]
    T_end: float = 0.10        # czas symulacji [s]

    @property
    def fs_lpf(self) -> float:
        """Czestotliwosc probkowania filtra = 1 / T_s_ctrl."""
        return 1.0 / self.controller.T_s_ctrl

    @property
    def N_per_ctrl(self) -> int:
        """Liczba krokow fizyki na jeden krok sterownika."""
        return int(round(self.controller.T_s_ctrl / self.dt_phys))

    @property
    def N_steps(self) -> int:
        """Calkowita liczba krokow fizyki."""
        return int(self.T_end / self.dt_phys)


def default_config() -> SimulationConfig:
    """Domyslna konfiguracja = stan PSIM dla Kroku 2.6."""
    return SimulationConfig(
        converter=ConverterParams(),
        controller=ControllerParams(),
    )
