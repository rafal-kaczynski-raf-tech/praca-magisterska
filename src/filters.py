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
    """Cyfrowy filtr dolnoprzepustowy (LPF) 2-go rzędu Butterwortha (biquad).

    Filtr wygladza poszarpany sygnal (np. z estymatora pradu), tlumic szybkie
    drgania i szum przelaczania kluczy, a przepuszczajac powolne, prawdziwe zmiany.
    """

    def __init__(self, fc: float, fs: float) -> None:
        # Wagi (wspolczynniki filtra) wyliczone dla zadanej czestotliwosci odciecia fc i probkowania fs
        # b0, b1, b2 - wagi dla probek wejsciowych (nowej i poprzednich)
        # a1, a2     - wagi dla poprzednich wyjsc filtra (pamiec wygladzona)
        self.b0, self.b1, self.b2, self.a1, self.a2 = butter2_lpf_coeffs(fc, fs)

        # Pamiec dwoch poprzednich wejsc (surowych probek x sprzed 1 i 2 krokow):
        self.x1: float = 0.0  # x[n-1]
        self.x2: float = 0.0  # x[n-2]

        # Pamiec dwoch poprzednich wyjsc (wygladzonych wynikow y sprzed 1 i 2 krokow):
        self.y1: float = 0.0  # y[n-1]
        self.y2: float = 0.0  # y[n-2]

    def process(self, x: float) -> float:
        """Przetwarza jedna nowa probke wejsciowa 'x' i zwraca wygladzona wartosc 'y'."""
        # 1. Rownanie roznicowe filtra:
        #    Bierzemy wazona sume obecnego i poprzednich wejsc (czesc 'b')
        #    oraz odejmujemy wazona sume poprzednich wygladzonych wynikow (czesc 'a').
        y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
             - self.a1 * self.y1 - self.a2 * self.y2)

        # 2. Przesuniecie pamieci o jeden krok w przeszlosc (na kolejny takt):
        self.x2 = self.x1  # to, co bylo 1 krok temu, staje sie stanem sprzed 2 krokow
        self.x1 = x        # obecna probka staje sie probka sprzed 1 kroku
        self.y2 = self.y1  # to samo dla wyjsc filtra
        self.y1 = y        # obecny wygladzony wynik staje sie wynikiem sprzed 1 kroku

        return y
