"""
Krok 2.6: MF-BB sterownik Z ESTYMACJĄ prądu obciążenia (BB with estimation).

Implementacja 1:1 wg pseudokodu PSIM C-block przekazanego przez prof. Iwańskiego
oraz parametrów z analizy danych referencyjnych z PSIM (psim_bb_estim_wyniki.csv).

Kluczowe różnice vs poprzedni "BB without estimation" (Krok 2.5):
  - estymator prądu obciążenia: iout_est = (1-s_old)*0.5*(i_act+i_old) - C*Δu/Ts
  - filtr LPF 2. rzędu Butterworth (fc=2 kHz @ fs=100 kHz) na iout_est
  - aktywna funkcja przełączająca: error_i = i_des - i_act, wi=1
  - i_des = iout_filt * u_ref / uin (bilans mocy)
  - R_load = 1000 Ω (R32 w schemacie - zweryfikowane V/I = 999.99 z CSV)
  - u_ref = 160 V (zweryfikowane z CSV)

Skrypt jest "samodzielny" - puszcza tylko symulację Pythona, zapisuje PNG.
Walidację 1:1 vs PSIM robi `porownanie_bb_estim_psim.py`.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
#  PARAMETRY OBWODU (z analizy CSV PSIM bb_WYNIKI.csv)
# =========================================================================
V_in    = 100.0      # napięcie źródła [V]
L       = 0.75e-3    # indukcyjność cewki [H]
R_L     = 0.05       # ESR cewki (R8 w schemacie) [Ω]
C       = 200e-6     # pojemność kondensatora [F]
# R_load: w pełnym schemacie BB-with-estimation R32(1kΩ) || R33(50Ω) = 47.619Ω
# Potwierdzone z CSV: udc/iout_filt = 239.6/4.99 = 48.1Ω, z bilansu mocy iL*(1-D) = 48.7Ω
R_load  = 47.619     # [Ω]

u_ref   = 240.0      # napięcie zadane [V]

# Warunki początkowe (z PSIM - C12 Initial V = 100 V)
v_C0    = 100.0
i_L0    = 0.0


# =========================================================================
#  PARAMETRY STEROWNIKA (z pseudokodu C-block PSIM + analizy schematu)
# =========================================================================
T_s_ctrl = 10e-6     # okres próbkowania sterownika [s] (= 1/100kHz / wew. = 50kHz*2)
wi       = 1.0       # waga błędu prądu w funkcji przełączającej
i_max    = 20.0      # zabezpieczenie nadprądowe [A]

# LPF 2. rząd Butterworth na iout_est (blok S19 w schemacie PSIM)
fc_lpf   = 2000.0    # częstotliwość odcięcia [Hz]
fs_lpf   = 1.0/T_s_ctrl   # 100 kHz - pracuje w takcie sterownika


# =========================================================================
#  KROK FIZYKI
# =========================================================================
# PSIM eksportuje co 0.5 µs - więc Time Step PSIM = 0.5 µs.
# Sterownik decyduje co 10 µs (= 20 kroków fizyki).
dt_phys  = 0.5e-6
N_per_ctrl = int(round(T_s_ctrl / dt_phys))   # 20

T_END    = 0.10      # czas symulacji [s] (pełny zakres CSV PSIM = 100 ms)
N_steps  = int(T_END / dt_phys)


# =========================================================================
#  Butterworth 2nd order LPF (bilinear transform, ręcznie - bez scipy)
# =========================================================================
def butter2_lpf_coeffs(fc: float, fs: float):
    """Współczynniki biquadu LPF 2-go rzędu Butterworth.

    Zwraca (b0, b1, b2, a1, a2) gdzie a0 = 1.
    Zastosowanie:
        y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    """
    Ts_lpf = 1.0 / fs
    # Pre-warp częstotliwości odcięcia (bilinear)
    omega_d = (2.0 / Ts_lpf) * np.tan(np.pi * fc / fs)
    K = 2.0 / Ts_lpf
    sqrt2 = np.sqrt(2.0)

    # H(s) = ωd² / (s² + √2·ωd·s + ωd²)
    den0 = K*K + sqrt2*omega_d*K + omega_d*omega_d   # współczynnik z²
    a0   = den0
    a1_r = (-2.0*K*K + 2.0*omega_d*omega_d) / a0
    a2_r = (K*K - sqrt2*omega_d*K + omega_d*omega_d) / a0
    b0   = (omega_d*omega_d) / a0
    b1   = 2.0 * b0
    b2   = b0
    return b0, b1, b2, a1_r, a2_r


B0, B1, B2, A1, A2 = butter2_lpf_coeffs(fc_lpf, fs_lpf)


# =========================================================================
#  SYMULACJA
# =========================================================================
def simulate():
    # Bufory pełnej długości
    t_arr  = np.arange(N_steps + 1) * dt_phys
    iL_arr = np.zeros(N_steps + 1)
    vC_arr = np.zeros(N_steps + 1)
    s_arr  = np.zeros(N_steps + 1, dtype=np.int8)

    # Zmienne logowane co krok sterownika (do debugu)
    n_ctrl = N_steps // N_per_ctrl + 1
    t_ctrl_arr        = np.zeros(n_ctrl)
    iout_est_arr      = np.zeros(n_ctrl)
    iout_filt_arr     = np.zeros(n_ctrl)
    i_des_arr         = np.zeros(n_ctrl)
    error_u_arr       = np.zeros(n_ctrl)
    error_i_arr       = np.zeros(n_ctrl)
    iL_sample_arr     = np.zeros(n_ctrl)   # widziane przez sterownik (z ZOH delay)
    vC_sample_arr     = np.zeros(n_ctrl)

    # Stan
    i_L = i_L0
    v_C = v_C0
    s   = 0      # PWM startuje od 0 (jak PSIM)
    iL_arr[0] = i_L
    vC_arr[0] = v_C
    s_arr[0]  = s

    # Pamięć sterownika (static w C-block)
    i_old      = 0.0
    s_old      = 0
    uout_old   = v_C0
    iout_filt  = 0.0
    # Stany filtra biquad (poprzednie wejścia/wyjścia)
    x1 = 0.0; x2 = 0.0   # iout_est[n-1], iout_est[n-2]
    y1 = 0.0; y2 = 0.0   # iout_filt[n-1], iout_filt[n-2]

    # Pamięć ZOH 1-sample delay (jak w Kroku 2.5 - sterownik widzi i_L(t-Ts), v_C(t-Ts))
    i_L_prev_sample = i_L0
    v_C_prev_sample = v_C0

    ctrl_idx = 0

    for k in range(1, N_steps + 1):
        t_now = k * dt_phys

        # Czy nadszedł krok sterownika (co 10 µs)?
        if (k - 1) % N_per_ctrl == 0:
            # Wartości "widziane" przez sterownik = poprzednia próbka (ZOH)
            i_act    = i_L_prev_sample
            uout_act = v_C_prev_sample

            # Estymator prądu obciążenia (tylko po t > 2*Ts, jak w pseudokodzie)
            if t_now > 2.0 * T_s_ctrl:
                iout_est = (1.0 - s_old) * 0.5 * (i_act + i_old) \
                           - C * (uout_act - uout_old) / T_s_ctrl
            else:
                iout_est = 0.0

            # LPF 2. rząd Butterworth (blok S19 w PSIM)
            iout_filt = B0*iout_est + B1*x1 + B2*x2 - A1*y1 - A2*y2
            x2 = x1; x1 = iout_est
            y2 = y1; y1 = iout_filt

            # Pożądany prąd cewki (bilans mocy)
            i_des = iout_filt * u_ref / V_in

            # Funkcja przełączająca
            error_u = u_ref - uout_act
            error_i = i_des - i_act
            s_new = 1 if (error_u + wi * error_i) > 0 else 0

            # Zabezpieczenie nadprądowe
            if i_act > i_max:
                s_new = 0
            elif i_act < -i_max:
                s_new = 1

            # Update pamięci sterownika
            s_old    = s
            uout_old = uout_act
            i_old    = i_act
            s        = s_new

            # Log
            t_ctrl_arr[ctrl_idx]    = t_now
            iout_est_arr[ctrl_idx]  = iout_est
            iout_filt_arr[ctrl_idx] = iout_filt
            i_des_arr[ctrl_idx]     = i_des
            error_u_arr[ctrl_idx]   = error_u
            error_i_arr[ctrl_idx]   = error_i
            iL_sample_arr[ctrl_idx] = i_act
            vC_sample_arr[ctrl_idx] = uout_act
            ctrl_idx += 1

            # Po skoku sterownika zapisujemy nową próbkę dla następnego ZOH
            i_L_prev_sample = i_L
            v_C_prev_sample = v_C

        # Fizyka (Euler, dt_phys = 0.5 µs)
        # synchroniczny boost: v_L = V_in - (1-s)*v_C - R_L*i_L
        #                       i_C = (1-s)*i_L - v_C/R_load
        v_L = V_in - (1.0 - s) * v_C - R_L * i_L
        i_C = (1.0 - s) * i_L - v_C / R_load
        i_L = i_L + v_L / L * dt_phys
        v_C = v_C + i_C / C * dt_phys

        iL_arr[k] = i_L
        vC_arr[k] = v_C
        s_arr[k]  = s

    return {
        "t":          t_arr,
        "i_L":        iL_arr,
        "v_C":        vC_arr,
        "s":          s_arr,
        "t_ctrl":     t_ctrl_arr[:ctrl_idx],
        "iout_est":   iout_est_arr[:ctrl_idx],
        "iout_filt":  iout_filt_arr[:ctrl_idx],
        "i_des":      i_des_arr[:ctrl_idx],
        "error_u":    error_u_arr[:ctrl_idx],
        "error_i":    error_i_arr[:ctrl_idx],
        "iL_sample":  iL_sample_arr[:ctrl_idx],
        "vC_sample":  vC_sample_arr[:ctrl_idx],
    }


def steady_state_metrics(res, t_start=0.20, t_end=0.29):
    """Średnie/ripple/duty w oknie steady-state (analogicznie do okna PSIM)."""
    t = res["t"]
    mask = (t >= t_start) & (t <= t_end)
    iL = res["i_L"][mask]
    vC = res["v_C"][mask]
    s  = res["s"][mask]

    print(f"\n=== Stan ustalony Python ({t_start*1e3:.0f}–{t_end*1e3:.0f} ms) ===")
    print(f"  udc:  mean={vC.mean():.3f} V   pp={vC.max()-vC.min():.3f} V   std={vC.std():.4f}")
    print(f"  iL:   mean={iL.mean():.3f} A   std={iL.std():.3f} A   "
          f"min={iL.min():.2f}  max={iL.max():.2f}")
    print(f"  duty: {s.mean():.4f}")

    # iout_filt z toku sterownika
    tc = res["t_ctrl"]
    mc = (tc >= t_start) & (tc <= t_end)
    print(f"  iout_filt (estymator): mean={res['iout_filt'][mc].mean():.4f} A   "
          f"i_des: mean={res['i_des'][mc].mean():.4f} A")


def plot_results(res, fname="wykres_bb_estim_python.png"):
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    ax[0].plot(res["t"]*1e3, res["v_C"], label="v_C (Python)", lw=0.6)
    ax[0].axhline(u_ref, color="r", ls="--", lw=0.8, label=f"u_ref={u_ref}V")
    ax[0].set_ylabel("Napięcie [V]")
    ax[0].set_title("BB with estimation — symulacja Python (R=1kΩ, u_ref=160V)")
    ax[0].grid(alpha=0.3); ax[0].legend(loc="lower right")

    ax[1].plot(res["t"]*1e3, res["i_L"], label="i_L (Python)", lw=0.4)
    ax[1].plot(res["t_ctrl"]*1e3, res["i_des"], label="i_des", color="orange", lw=0.8)
    ax[1].set_ylabel("Prąd cewki [A]")
    ax[1].grid(alpha=0.3); ax[1].legend(loc="upper right")

    ax[2].plot(res["t_ctrl"]*1e3, res["iout_est"], label="iout_est", lw=0.5, alpha=0.6)
    ax[2].plot(res["t_ctrl"]*1e3, res["iout_filt"], label="iout_filt (LPF 2kHz)",
               lw=1.0, color="red")
    ax[2].axhline(u_ref/R_load, color="green", ls="--", lw=0.8,
                  label=f"i_load_true={u_ref/R_load:.4f}A")
    ax[2].set_ylabel("Estymata I_out [A]")
    ax[2].grid(alpha=0.3); ax[2].legend(loc="upper right")

    ax[3].step(res["t"]*1e3, res["s"], where="post", lw=0.5)
    ax[3].set_ylabel("PWM s")
    ax[3].set_xlabel("Czas [ms]")
    ax[3].grid(alpha=0.3); ax[3].set_ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    print(f"\nZapisano wykres: {fname}")


if __name__ == "__main__":
    print(f"Butterworth 2nd order LPF coeffs (fc={fc_lpf}Hz, fs={fs_lpf:.0f}Hz):")
    print(f"  b = ({B0:.6e}, {B1:.6e}, {B2:.6e})")
    print(f"  a = (1, {A1:.6e}, {A2:.6e})")
    print(f"\nSymulacja: T={T_END*1e3:.0f}ms, dt_phys={dt_phys*1e6}µs, "
          f"T_s_ctrl={T_s_ctrl*1e6}µs, kroków={N_steps}")

    res = simulate()
    steady_state_metrics(res, t_start=0.080, t_end=0.100)
    plot_results(res)
