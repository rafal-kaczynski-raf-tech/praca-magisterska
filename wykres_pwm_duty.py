"""
wykres_pwm_duty.py — współczynnik wypełnienia PWM w oknie ruchomym (Krok 2.5).

Zamiast pokazywać surowe szpilki PWM (które rozjeżdżają się fazowo w stanie
ustalonym z powodu chaosu deterministycznego sterowania histerezowego),
ten wykres prezentuje średni współczynnik wypełnienia (duty cycle) wyliczany
w oknie ruchomym o szerokości WINDOW_MS = 1.0 ms (≈100 cykli sterownika przy
T_s_ctrl=10 µs). Uśrednienie redukuje wpływ chaosu histerezowego — w stanie
ustalonym widoczne są jeszcze oscylacje (PSIM ±0.04, Python ±0.08), ale
**średnia długoterminowa jest praktycznie identyczna** (różnica ~0.7%).
Faza narastania (0–3 ms) pokrywa się dokładnie.

Kod symulacji Python skopiowany z porownanie_bb_psim.py — gwarantowana
zgodność z wykresem finalnym.

Wynik: wykres_pwm_duty.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==============================================================================
# 1. WCZYTANIE DANYCH Z PSIM
# ==============================================================================

psim_data = np.loadtxt('psim_bb_wyniki.txt', skiprows=1)

psim_t = psim_data[:, 0]
psim_pwm = psim_data[:, 7]

# ==============================================================================
# 2. PARAMETRY (identyczne z porownanie_bb_psim.py)
# ==============================================================================

V_in = 100.0
L = 0.75e-3
R_L = 0.05
C = 0.2e-3
R_load = 50.0

u_ref = 240.0
wi = 0.5
i_max = 20.0
alpha = 0.01
beta = 0.98

t_end = 0.010
dt = 10e-6
T_s_ctrl = 10e-6
ctrl_decimation = int(T_s_ctrl / dt)

WINDOW_MS = 1.0  # szerokość okna ruchomego dla duty cycle [ms]
                 # 0.2 ms — za krótkie, chaos jeszcze dominuje
                 # 1.0 ms — uśrednia ~100 cykli sterownika, średnie się zgadzają
                 #          w stanie ustalonym, drobne oscylacje wciąż widoczne
                 # 2.0+ ms — zniekształca dynamikę stanu przejściowego

# ==============================================================================
# 3. SYMULACJA PYTHON (kopia z porownanie_bb_psim.py)
# ==============================================================================

t_vals = np.arange(0, t_end, dt)
n_steps = len(t_vals)

S_vals = np.zeros(n_steps)

i_L = 0.0
v_C = 100.0
i_exp = 0.0
i_old = 0.0
i_exp_old = 0.0
s = 0.0

i_L_prev_sample = 0.0
v_C_prev_sample = 100.0


def derivs(iL, vC, s_val):
    if s_val == 1.0:
        diL = (V_in - iL * R_L) / L
        dvC = (-vC / R_load) / C
    else:
        diL = (V_in - iL * R_L - vC) / L
        dvC = (iL - vC / R_load) / C
    return diL, dvC


for k in range(1, n_steps):
    if k % ctrl_decimation == 0:
        i_act = i_L_prev_sample
        uout_act = v_C_prev_sample
        i_L_prev_sample = i_L
        v_C_prev_sample = v_C

        error_u = (1.0 + wi * i_act / u_ref) * u_ref - uout_act
        i_exp = alpha * i_act + alpha * i_old + beta * i_exp_old
        i_old = i_act
        i_exp_old = i_exp
        error_i = 0 * i_exp - i_act
        s = 1.0 if (error_u + wi * error_i) > 0 else 0.0
        if i_act > i_max:
            s = 0.0
        if i_act < -i_max:
            s = 1.0

    diL, dvC = derivs(i_L, v_C, s)
    i_L = i_L + diL * dt
    v_C = v_C + dvC * dt
    S_vals[k] = s

# ==============================================================================
# 4. DUTY CYCLE W OKNIE RUCHOMYM
# ==============================================================================

def moving_duty(pwm_signal, t_signal, window_ms):
    """Średnia wartość PWM w oknie [t-W/2, t+W/2] dla każdego punktu t."""
    if len(t_signal) < 2:
        return np.zeros_like(pwm_signal)
    dt_local = np.median(np.diff(t_signal))
    n_window = max(1, int(round((window_ms * 1e-3) / dt_local)))
    kernel = np.ones(n_window) / n_window
    return np.convolve(pwm_signal, kernel, mode='same')

psim_duty = moving_duty(psim_pwm, psim_t, WINDOW_MS)
py_duty = moving_duty(S_vals, t_vals, WINDOW_MS)

# ==============================================================================
# 5. WYKRES
# ==============================================================================

t_ms = t_vals * 1000
psim_t_ms = psim_t * 1000

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

ax.plot(psim_t_ms, psim_duty, 'r', linewidth=2.0, label='PSIM — duty cycle', alpha=0.85)
ax.plot(t_ms, py_duty, 'b--', linewidth=1.5, label='Python — duty cycle', alpha=0.85)

ax.set_xlabel('Czas [ms]')
ax.set_ylabel('Duty cycle [-]')
ax.set_title(
    f'Współczynnik wypełnienia PWM (okno ruchome {WINDOW_MS} ms) — Python vs PSIM\n'
    f'L={L*1e3}mH, C={C*1e6:.0f}µF, R={R_load}Ω, u_ref={u_ref}V'
)
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.05, 1.05])
ax.set_xlim([0, t_end * 1000])

plt.tight_layout()
plt.savefig('wykres_pwm_duty.png', dpi=150, bbox_inches='tight')
print("Zapisano: wykres_pwm_duty.png")

# Metryki uśrednione
print("\n" + "=" * 60)
print("DUTY CYCLE — Python vs PSIM (okno ruchome)")
print("=" * 60)
psim_duty_interp = np.interp(t_vals, psim_t, psim_duty)
duty_mae = np.mean(np.abs(psim_duty_interp - py_duty))
duty_rmse = np.sqrt(np.mean((psim_duty_interp - py_duty) ** 2))
print(f"MAE  duty (pełna symulacja):  {duty_mae:.4f}")
print(f"RMSE duty (pełna symulacja):  {duty_rmse:.4f}")

ss_mask = t_vals > 0.006
print(f"\nStan ustalony (t > 6 ms):")
print(f"  Średni duty PSIM:   {np.mean(psim_duty_interp[ss_mask]):.4f}")
print(f"  Średni duty Python: {np.mean(py_duty[ss_mask]):.4f}")
print(f"  Różnica:            {np.mean(py_duty[ss_mask]) - np.mean(psim_duty_interp[ss_mask]):+.4f}")
print("=" * 60)
