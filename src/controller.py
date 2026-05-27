"""Sterownik MF-BB z estymacja pradu obciazenia (Krok 2.6).

Algorytm 1:1 z pseudokodu PSIM C-block prof. Iwanskiego.

Sekwencja w jednym ticku sterownika:
    1. Pobierz pomiary (i_act, uout_act) z ZOH samplera
    2. Estymator pradu obciazenia (po fazie startowej):
         iout_est = (1 - s_old) * 0.5*(i_act + i_old)
                    - C * (uout_act - uout_old) / Ts
    3. Filtr LPF 2-go rzedu Butterwortha
    4. Prad zadany: i_des = iout_filt * u_ref / V_in (bilans mocy)
    5. Decyzja kluczy:
         s_new = 1 if (u_ref - uout_act) + wi*(i_des - i_act) > 0 else 0
    6. Zabezpieczenie nadpradowe
    7. Update pamieci (i_old, s_old, uout_old)
"""
from __future__ import annotations
from dataclasses import dataclass
from .config import ControllerParams, ConverterParams
from .filters import ButterworthLPF


@dataclass
class ControllerOutput:
    """Pelny output jednego ticku sterownika (do logowania)."""
    s_new: int
    iout_est: float
    iout_filt: float
    i_des: float
    error_u: float
    error_i: float
    i_act: float
    uout_act: float


class MFBBController:
    """Sterownik bezmodelowy Bang-Bang z estymatorem pradu obciazenia."""

    def __init__(
        self,
        params: ControllerParams,
        converter_params: ConverterParams,
        fs_lpf: float,
    ) -> None:
        self.p = params
        self.C = converter_params.C       # potrzebne w estymatorze
        self.V_in = converter_params.V_in  # potrzebne w bilansie mocy
        self.lpf = ButterworthLPF(params.fc_lpf, fs_lpf)

        # Pamiec sterownika (odpowiednik "static" w C-block)
        self.s_current: int = 0
        self.i_old: float = 0.0
        self.s_old: int = 0
        self.uout_old: float = converter_params.v_C0
        self.u_ref: float = params.u_ref  # mutable - moze byc zmienione w runtime (scenariusz)

    def tick(self, t_now: float, i_act: float, uout_act: float) -> ControllerOutput:
        """Jeden krok sterownika.  Zwraca dane do logowania, aktualizuje stan wewn."""
        p = self.p

        # Estymator prądu obciażenia (po fazie startowej 2*Ts)
        if t_now > 2.0 * p.T_s_ctrl:
            iout_est = (1.0 - self.s_old) * 0.5 * (i_act + self.i_old) \
                       - self.C * (uout_act - self.uout_old) / p.T_s_ctrl
        else:
            iout_est = 0.0

        # LPF 2-go rzędu
        iout_filt = self.lpf.process(iout_est)

        # Bilans mocy - prąd zadany
        i_des = iout_filt * self.u_ref / self.V_in

        # Funkcja przelaczajaca
        error_u = self.u_ref - uout_act
        error_i = i_des - i_act
        s_new = 1 if (error_u + p.wi * error_i) > 0 else 0

        # Zabezpieczenie nadpradowe
        if i_act > p.i_max:
            s_new = 0
        elif i_act < -p.i_max:
            s_new = 1

        # Update pamieci
        self.s_old = self.s_current
        self.uout_old = uout_act
        self.i_old = i_act
        self.s_current = s_new

        return ControllerOutput(
            s_new=s_new,
            iout_est=iout_est,
            iout_filt=iout_filt,
            i_des=i_des,
            error_u=error_u,
            error_i=error_i,
            i_act=i_act,
            uout_act=uout_act,
        )
