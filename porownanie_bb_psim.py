"""
porownanie_bb_psim.py — nakładanie przebiegów Python (MF-BB) na referencję z PSIM.

Wczytuje psim_bb_wyniki.txt (9 kolumn: Time, I(L7), i_out_M_BB1, udc_BB1, i_pred_BB1,
iL_BB1, Uref_BB1, pwmBB1, i_ref_BB1), uruchamia identyczny algorytm MF-BB w Pythonie
z parametrami fizycznymi PSIM (R_L=0.05Ω, ESR cewki) i generuje wykres porównawczy
zapisywany do wykres_bb_psim_comparison.png.

Backend Matplotlib ustawiony na "Agg" — działa bez wyświetlania okna (CI/headless).
Mierzy MAE prądu cewki i napięcia wyjściowego jako liczbową miarę zgodności
(Krok 2.5: ~2.3 V napięcia — patrz CLAUDE_CONTEXT.md).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (zapisuje plik bez okna)
import matplotlib.pyplot as plt

# ==============================================================================
# 1. WCZYTANIE DANYCH Z PSIM
# ==============================================================================

psim_data = np.loadtxt('psim_bb_wyniki.txt', skiprows=1)

psim_t = psim_data[:, 0]          # Time [s]
psim_iL = psim_data[:, 1]         # I(L7) - prąd cewki (col 1)
psim_udc = psim_data[:, 3]        # udc_BB1 - napięcie wyjściowe (col 3)
psim_uref = psim_data[:, 6]       # Uref_BB1 - napięcie referencyjne (col 6)
psim_pwm = psim_data[:, 7]        # pwmBB1 - sygnał przełączający (col 7)
psim_iref = psim_data[:, 8]       # i_ref_BB1 - prąd referencyjny (col 8)

# ==============================================================================
# 2. PARAMETRY FIZYCZNE (identyczne jak PSIM)
# ==============================================================================

V_in = 100.0
L = 0.75e-3
R_L = 0.05           # R8 = 0.05 Ω (ESR cewki, w szereg z L7)
C = 0.2e-3
R_load = 50.0       # R33 (R32=1kΩ jest w obwodzie sterowania, NIE w obwodzie mocy)

u_ref = 240.0
wi = 0.5
i_max = 20.0
alpha = 0.01
beta = 0.98

# ==============================================================================
# 3. SYMULACJA PYTHON (algorytm identyczny z PSIM C-block)
# ==============================================================================

t_end = 0.010
dt = 10e-6        # 10µs - taki sam jak PSIM Time Step (i T_s_ctrl)
T_s_ctrl = 10e-6
ctrl_decimation = int(T_s_ctrl / dt)  # = 1 (sterownik co każdy krok)

t_vals = np.arange(0, t_end, dt)
n_steps = len(t_vals)

i_L_vals = np.zeros(n_steps)
v_C_vals = np.zeros(n_steps)
i_exp_vals = np.zeros(n_steps)
S_vals = np.zeros(n_steps)

i_L = 0.0
v_C = 100.0
i_exp = 0.0
i_old = 0.0
i_exp_old = 0.0
s = 0.0  # PSIM start: s=0 (sterownik jeszcze nie zadecydował)

# Buffer dla delay'u pomiaru prądu (PSIM iL_BB1(t) = I(L7)(t-Ts))
i_L_prev_sample = 0.0
v_C_prev_sample = 100.0

i_L_vals[0] = i_L
v_C_vals[0] = v_C


def derivs(iL, vC, s_val):
    """Pochodne stanu dla danego s."""
    if s_val == 1.0:
        # Tryb 1: ON (ładuj cewkę)
        diL = (V_in - iL * R_L) / L
        dvC = (-vC / R_load) / C
    else:
        # Tryb 2: OFF (boost)
        diL = (V_in - iL * R_L - vC) / L
        dvC = (iL - vC / R_load) / C
    return diL, dvC


for k in range(1, n_steps):

    # KONWENCJA PSIM: arr[k] = stan AT t=k*dt (PRZED fizyką dla nast. interwału)
    # Sterownik najpierw decyduje s na podstawie BIEŻĄCEGO stanu (sampluje x1..x4)
    # Potem fizyka liczy nowy stan dla [k*dt, (k+1)*dt]

    # 1) Sterownik z OPÓŹNIONYM stanem (PSIM ZOH 1-sample delay)
    if k % ctrl_decimation == 0:
        i_act = i_L_prev_sample      # PSIM: iL_BB1(t) = I(L7)(t-Ts)
        uout_act = v_C_prev_sample   # tak samo dla napięcia

        # Zapisz BIEŻĄCY stan jako "poprzednią próbkę" dla następnego cyklu
        i_L_prev_sample = i_L
        v_C_prev_sample = v_C

        # PSIM linia 32
        error_u = (1.0 + wi * i_act / u_ref) * u_ref - uout_act

        # PSIM linia 35: filtr LP
        i_exp = alpha * i_act + alpha * i_old + beta * i_exp_old
        i_old = i_act
        i_exp_old = i_exp

        # PSIM linia 41: error_i (0*i_exp)
        error_i = 0 * i_exp - i_act

        # PSIM linia 43: przełączanie
        s = 1.0 if (error_u + wi * error_i) > 0 else 0.0

        # PSIM linia 45-48: zabezpieczenie nadprądowe
        if i_act > i_max:
            s = 0.0
        if i_act < -i_max:
            s = 1.0

    # 2) Zapisz BIEŻĄCY stan (PRZED fizyką) - jak PSIM
    i_L_vals[k] = i_L
    v_C_vals[k] = v_C
    i_exp_vals[k] = i_exp

    # 3) Fizyka - Euler dla [k*dt, (k+1)*dt]
    diL, dvC = derivs(i_L, v_C, s)
    i_L = i_L + diL * dt
    v_C = v_C + dvC * dt
    S_vals[k] = s

# ==============================================================================
# 4. WYKRESY - NAŁOŻENIE PYTHON NA PSIM
# ==============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

t_ms = t_vals * 1000
psim_t_ms = psim_t * 1000

# --- Napięcie wyjściowe ---
axes[0].plot(psim_t_ms, psim_udc, 'r', linewidth=1.5, label='PSIM (udc_BB1)', alpha=0.8)
axes[0].plot(t_ms, v_C_vals, 'b--', linewidth=1, label='Python (v_C)', alpha=0.8)
axes[0].axhline(y=u_ref, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'u_ref = {u_ref}V')
axes[0].set_ylabel('Napięcie [V]')
axes[0].set_title('Napięcie wyjściowe — PSIM vs Python')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# --- Prąd cewki ---
axes[1].plot(psim_t_ms, psim_iL, 'r', linewidth=1, label='PSIM I(L7)', alpha=0.8)
axes[1].plot(t_ms, i_L_vals, 'b--', linewidth=0.8, label='Python i_L', alpha=0.8)
axes[1].plot(psim_t_ms, psim_iref, 'm', linewidth=1, label='PSIM i_ref (i_exp)', alpha=0.6)
axes[1].plot(t_ms, i_exp_vals, 'c--', linewidth=0.8, label='Python i_exp', alpha=0.6)
axes[1].set_ylabel('Prąd [A]')
axes[1].set_title('Prąd cewki — PSIM vs Python')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# --- Sygnał przełączający ---
axes[2].plot(psim_t_ms, psim_pwm, 'r', linewidth=0.8, label='PSIM pwmBB1', alpha=0.7)
axes[2].plot(t_ms, S_vals, 'b--', linewidth=0.5, label='Python S', alpha=0.7)
axes[2].set_ylabel('PWM [-]')
axes[2].set_xlabel('Czas [ms]')
axes[2].set_title('Sygnał przełączający — PSIM vs Python')
axes[2].legend(loc='upper right')
axes[2].set_ylim([-0.1, 1.1])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(
    'Walidacja Digital Twin: Python vs PSIM\n'
    f'BB without load current estimation — L={L*1e3}mH, C={C*1e6:.0f}µF, R={R_load}Ω, u_ref={u_ref}V',
    fontsize=13, fontweight='bold', y=1.03
)
plt.savefig('porownanie_bb_python_vs_psim.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# 5. METRYKI ZGODNOŚCI
# ==============================================================================

# Interpolacja Pythona do siatki czasowej PSIM
py_udc_interp = np.interp(psim_t, t_vals, v_C_vals)
py_iL_interp = np.interp(psim_t, t_vals, i_L_vals)

# Błędy
udc_error = psim_udc - py_udc_interp
iL_error = psim_iL - py_iL_interp

print("\n" + "=" * 65)
print("METRYKI ZGODNOŚCI: Python vs PSIM")
print("=" * 65)
print(f"Napięcie wyjściowe (udc) — cała symulacja:")
print(f"  MAE  = {np.mean(np.abs(udc_error)):.3f} V")
print(f"  RMSE = {np.sqrt(np.mean(udc_error**2)):.3f} V")
print(f"  MAX  = {np.max(np.abs(udc_error)):.3f} V")
print()

# Stan ustalony: ostatnie 4ms (po t=6ms)
ss_mask = psim_t > 0.006
print(f"Stan ustalony (t > 6ms):")
print(f"  Średnie napięcie: PSIM={np.mean(psim_udc[ss_mask]):.2f}V, "
      f"Python={np.mean(py_udc_interp[ss_mask]):.2f}V "
      f"(różnica: {np.mean(py_udc_interp[ss_mask]) - np.mean(psim_udc[ss_mask]):+.2f}V)")
print(f"  Tetnienia (peak-peak): PSIM={np.ptp(psim_udc[ss_mask]):.2f}V, "
      f"Python={np.ptp(py_udc_interp[ss_mask]):.2f}V")
print(f"  Średni prąd cewki: PSIM={np.mean(psim_iL[ss_mask]):.2f}A, "
      f"Python={np.mean(py_iL_interp[ss_mask]):.2f}A")
print()
print(f"Prąd cewki (i_L) — pełna symulacja:")
print(f"  MAE  = {np.mean(np.abs(iL_error)):.3f} A")
print(f"  RMSE = {np.sqrt(np.mean(iL_error**2)):.3f} A")
print("=" * 65)
