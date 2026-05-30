"""Filtry cyfrowe - implementacje 1:1 z proceduralnej wersji."""
from __future__ import annotations
import numpy as np


def butter2_lpf_coeffs(fc: float, fs: float) -> tuple[float, float, float, float, float]:
    """Wspolczynniki biquadu LPF 2-go rzedu Butterworth (bilinear transform).

    Zwraca (b0, b1, b2, a1, a2) gdzie a0 = 1.
    Forma roznicowa:
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    """
    Ts = 1.0 / fs
    omega_d = (2.0 / Ts) * np.tan(np.pi * fc / fs)  # pre-warp
    K = 2.0 / Ts
    sqrt2 = np.sqrt(2.0)
    den0 = K*K + sqrt2*omega_d*K + omega_d*omega_d
    a0 = den0
    a1 = (-2.0*K*K + 2.0*omega_d*omega_d) / a0
    a2 = (K*K - sqrt2*omega_d*K + omega_d*omega_d) / a0
    b0 = (omega_d*omega_d) / a0
    b1 = 2.0 * b0
    b2 = b0
    return b0, b1, b2, a1, a2


class ButterworthLPF:
    """Filtr LPF 2-go rzedu (biquad) z hermetyzowanym stanem."""

    def __init__(self, fc: float, fs: float) -> None:
        self.b0, self.b1, self.b2, self.a1, self.a2 = butter2_lpf_coeffs(fc, fs)
        # stan: poprzednie wejscia i wyjscia
        self.x1: float = 0.0
        self.x2: float = 0.0
        self.y1: float = 0.0
        self.y2: float = 0.0

    def process(self, x: float) -> float:
        """Jedno wywolanie filtra na probke wejsciowa x."""
        y = self.b0*x + self.b1*self.x1 + self.b2*self.x2 \
            - self.a1*self.y1 - self.a2*self.y2
        self.x2 = self.x1
        self.x1 = x
        self.y2 = self.y1
        self.y1 = y
        return y
