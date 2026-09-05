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
        N_steps = cfg.N_steps          # Liczba krokow fizyki (np. 600 000 dla 300 ms przy dt=0.5 us)
        N_per_ctrl = cfg.N_per_ctrl    # Ilosc krokow fizyki na 1 decyzje sterownika (np. 10 us / 0.5 us = 20)
        dt = cfg.dt_phys               # Krok calkowania fizyki (0.5 us)

        # ---------------------------------------------------------------------
        # 1. Prealokacja buforow fizyki (probkowanie geste: co dt = 0.5 us)
        # ---------------------------------------------------------------------
        t_arr = np.arange(N_steps + 1) * dt           # Os czasu: [0, 0.5us, 1.0us, ..., T_end]
        iL_arr = np.zeros(N_steps + 1)                # Rzeczywisty prad cewki w kazdym kroku fizyki
        vC_arr = np.zeros(N_steps + 1)                # Rzeczywiste napiecie kondensatora w kazdym kroku
        s_arr = np.zeros(N_steps + 1, dtype=np.int8)  # Stan klucza (0 lub 1) aktywny w danym kroku fizyki
        # i_des rozciagniete na siatke fizyki (ZOH - trzymane miedzy tickami sterownika dla funkcji celu)
        i_des_phys_arr = np.zeros(N_steps + 1)

        # ---------------------------------------------------------------------
        # 2. Prealokacja buforow sterownika (probkowanie rzadkie: co Ts = 10 us)
        # ---------------------------------------------------------------------
        n_ctrl = N_steps // N_per_ctrl + 1            # Liczba decyzji sterownika (np. 30 000)
        t_ctrl_arr = np.zeros(n_ctrl)                 # Chwile czasowe wykonania ticku sterownika
        iout_est_arr = np.zeros(n_ctrl)               # Surowy estymowany prad obciazenia (C * du/dt)
        iout_filt_arr = np.zeros(n_ctrl)              # Wygladzony prad obciazenia po filtrze LPF
        i_des_arr = np.zeros(n_ctrl)                  # Prad zadany cewki wyliczony z bilansu mocy
        error_u_arr = np.zeros(n_ctrl)                # Uchyb napiecia: u_ref - uout_act
        error_i_arr = np.zeros(n_ctrl)                # Uchyb pradu: i_des - i_act
        iL_sample_arr = np.zeros(n_ctrl)              # Prad pobrany przez sterownik z samplera
        vC_sample_arr = np.zeros(n_ctrl)              # Napiecie pobrane przez sterownik z samplera

        # ---------------------------------------------------------------------
        # 3. Zapis warunkow poczatkowych (krok 0)
        # ---------------------------------------------------------------------
        iL_arr[0] = self.converter.i_L
        vC_arr[0] = self.converter.v_C
        s_arr[0] = self.controller.s_current  # = 0

        ctrl_idx = 0
        i_des_current = 0.0   # Ostatnio wyznaczony i_des (trzymany miedzy tickami sterownika)

        # ---------------------------------------------------------------------
        # 4. Inicjalizacja obslugi zdarzen scenariusza (skoki R_load i u_ref)
        # ---------------------------------------------------------------------
        scn = cfg.scenario
        load_done = scn is None or scn.load_step_time is None
        ref_done = scn is None or scn.ref_step_time is None
        ref_pulses = None if scn is None else scn.ref_pulses
        pulse_idx = 0
        load_pulses = None if scn is None else scn.load_pulses
        load_pulse_idx = 0

        # ---------------------------------------------------------------------
        # 5. Glowna petla symulacji (krok fizyki k = 1 .. N_steps)
        # ---------------------------------------------------------------------
        for k in range(1, N_steps + 1):
            t_now = k * dt

            # -----------------------------------------------------------------
            # BLOK A: Tick sterownika (wykonywany co N_per_ctrl krokow, np. co 20 krokow = 10 us)
            # -----------------------------------------------------------------
            if (k - 1) % N_per_ctrl == 0:
                # 1. Sprawdz i zaaplikuj zdarzenia scenariusza (skok u_ref lub zmiana R_load)
                if not load_done and t_now >= scn.load_step_time:
                    self.converter.R_load = scn.load_step_R
                    load_done = True
                if not ref_done and t_now >= scn.ref_step_time:
                    self.controller.u_ref = scn.ref_step_value
                    ref_done = True
                if ref_pulses is not None:
                    while (pulse_idx < len(ref_pulses)
                           and t_now >= ref_pulses[pulse_idx][0]):
                        self.controller.u_ref = ref_pulses[pulse_idx][1]
                        pulse_idx += 1
                if load_pulses is not None:
                    while (load_pulse_idx < len(load_pulses)
                           and t_now >= load_pulses[load_pulse_idx][0]):
                        self.converter.R_load = load_pulses[load_pulse_idx][1]
                        load_pulse_idx += 1

                # 2. Odczyt z samplera (ZOH: wartosci zbuforowane z opoznieniem 1 cyklu)
                i_act, uout_act = self.sampler.read()

                # 3. Krok algorytmu sterownika: estymacja i_out, filtr LPF, bilans mocy, decyzja s_new
                out = self.controller.tick(t_now, i_act, uout_act)

                # 4. Zapis diagnostyczny do buforow sterownika
                t_ctrl_arr[ctrl_idx] = t_now
                iout_est_arr[ctrl_idx] = out.iout_est
                iout_filt_arr[ctrl_idx] = out.iout_filt
                i_des_arr[ctrl_idx] = out.i_des
                i_des_current = out.i_des
                error_u_arr[ctrl_idx] = out.error_u
                error_i_arr[ctrl_idx] = out.error_i
                iL_sample_arr[ctrl_idx] = i_act
                vC_sample_arr[ctrl_idx] = uout_act
                ctrl_idx += 1

                # 5. Zatrzaśnięcie obecnego stanu fizyki w samplerze na KOLEJNY tick sterownika
                self.sampler.latch(self.converter.i_L, self.converter.v_C)

            # -----------------------------------------------------------------
            # BLOK B: Krok fizyki (calkowanie Eulera w przod dla biezacego s)
            # -----------------------------------------------------------------
            self.converter.step(self.controller.s_current, dt)

            # -----------------------------------------------------------------
            # BLOK C: Zapis pelnego stanu fizyki do prealokowanych tablic
            # -----------------------------------------------------------------
            iL_arr[k] = self.converter.i_L
            vC_arr[k] = self.converter.v_C
            s_arr[k] = self.controller.s_current
            i_des_phys_arr[k] = i_des_current

        # ---------------------------------------------------------------------
        # 6. Zwrocenie kompletnego slownika wynikow (wszystkie przebiegi czasowe)
        # ---------------------------------------------------------------------
        return {
            "t": t_arr,
            "i_L": iL_arr,
            "v_C": vC_arr,
            "s": s_arr,
            "t_ctrl": t_ctrl_arr[:ctrl_idx],
            "iout_est": iout_est_arr[:ctrl_idx],
            "iout_filt": iout_filt_arr[:ctrl_idx],
            "i_des": i_des_arr[:ctrl_idx],
            "i_des_phys": i_des_phys_arr,
            "error_u": error_u_arr[:ctrl_idx],
            "error_i": error_i_arr[:ctrl_idx],
            "iL_sample": iL_sample_arr[:ctrl_idx],
            "vC_sample": vC_sample_arr[:ctrl_idx],
        }
