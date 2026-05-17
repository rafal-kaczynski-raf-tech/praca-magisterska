"""Orkiestrator symulacji MF-BB.

Pętla glowna: krok fizyki co dt_phys (0.5 us), krok sterownika co T_s_ctrl (10 us).
Logowanie do prealokowanych ndarray dla wydajnosci.

KOLEJNOSC OPERACJI w petli musi byc IDENTYCZNA z proceduralnym
demo_bangbang_estim.simulate() (sprawdza to tests/test_regression.py).
"""
from __future__ import annotations
import numpy as np
from .config import SimulationConfig
from .converter import Converter
from .controller import MFBBController
from .sampler import ZOHSampler


class Simulator:
    """Orkiestrator pelnej symulacji MF-BB."""

    def __init__(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.converter = Converter(config.converter)
        self.controller = MFBBController(
            params=config.controller,
            converter_params=config.converter,
            fs_lpf=config.fs_lpf,
        )
        self.sampler = ZOHSampler(
            i_L0=config.converter.i_L0,
            v_C0=config.converter.v_C0,
        )

    def run(self) -> dict:
        """Wykonuje pełna symulacje T_end sekund, zwraca slownik tablic."""
        cfg = self.cfg
        N_steps = cfg.N_steps
        N_per_ctrl = cfg.N_per_ctrl
        dt = cfg.dt_phys

        # Bufory fizyki
        t_arr = np.arange(N_steps + 1) * dt
        iL_arr = np.zeros(N_steps + 1)
        vC_arr = np.zeros(N_steps + 1)
        s_arr = np.zeros(N_steps + 1, dtype=np.int8)

        # Bufory sterownika
        n_ctrl = N_steps // N_per_ctrl + 1
        t_ctrl_arr = np.zeros(n_ctrl)
        iout_est_arr = np.zeros(n_ctrl)
        iout_filt_arr = np.zeros(n_ctrl)
        i_des_arr = np.zeros(n_ctrl)
        error_u_arr = np.zeros(n_ctrl)
        error_i_arr = np.zeros(n_ctrl)
        iL_sample_arr = np.zeros(n_ctrl)
        vC_sample_arr = np.zeros(n_ctrl)

        # Warunki poczatkowe -> probka 0
        iL_arr[0] = self.converter.i_L
        vC_arr[0] = self.converter.v_C
        s_arr[0] = self.controller.s_current  # = 0

        ctrl_idx = 0

        for k in range(1, N_steps + 1):
            t_now = k * dt

            # Tick sterownika co N_per_ctrl krokow fizyki (1:1 z proceduralna wersja)
            if (k - 1) % N_per_ctrl == 0:
                i_act, uout_act = self.sampler.read()
                out = self.controller.tick(t_now, i_act, uout_act)

                t_ctrl_arr[ctrl_idx] = t_now
                iout_est_arr[ctrl_idx] = out.iout_est
                iout_filt_arr[ctrl_idx] = out.iout_filt
                i_des_arr[ctrl_idx] = out.i_des
                error_u_arr[ctrl_idx] = out.error_u
                error_i_arr[ctrl_idx] = out.error_i
                iL_sample_arr[ctrl_idx] = i_act
                vC_sample_arr[ctrl_idx] = uout_act
                ctrl_idx += 1

                # Zatrzasniecie nowej probki dla nastepnego ticku (po decyzji)
                self.sampler.latch(self.converter.i_L, self.converter.v_C)

            # Krok fizyki - uzywa AKTUALNEJ decyzji klucza
            self.converter.step(self.controller.s_current, dt)

            iL_arr[k] = self.converter.i_L
            vC_arr[k] = self.converter.v_C
            s_arr[k] = self.controller.s_current

        return {
            "t": t_arr,
            "i_L": iL_arr,
            "v_C": vC_arr,
            "s": s_arr,
            "t_ctrl": t_ctrl_arr[:ctrl_idx],
            "iout_est": iout_est_arr[:ctrl_idx],
            "iout_filt": iout_filt_arr[:ctrl_idx],
            "i_des": i_des_arr[:ctrl_idx],
            "error_u": error_u_arr[:ctrl_idx],
            "error_i": error_i_arr[:ctrl_idx],
            "iL_sample": iL_sample_arr[:ctrl_idx],
            "vC_sample": vC_sample_arr[:ctrl_idx],
        }
