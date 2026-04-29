"""
demo_bangbang_psim.py — replikacja sterownika MF-BB 1:1 wg PSIM (Krok 2.5).

Wersja zwalidowana z C-blockiem PSIM: identyczny algorytm bang-bang, identyczne
parametry filtru LP (alpha=0.01, beta=0.98), R_L=0 (PSIM domyślnie bez ESR cewki),
T_s_ctrl=10 µs, u_ref=240 V (stałe). Cel: dopasować przebiegi Pythona do PSIM
możliwie dokładnie, jako punkt odniesienia dla późniejszych eksperymentów PSO.

Plik samodzielnie uruchamia symulację — nie czyta danych PSIM (zostawione to
porownanie_bb_psim.py). Wyniki należy zapisać i nałożyć osobnym narzędziem.
"""
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARAMETRY FIZYCZNE UKŁADU (dokładnie jak w PSIM)
# ==============================================================================

V_in = 100.0        # Napięcie wejściowe [V]
L = 0.75e-3         # Indukcyjność [H] - 0.75 mH
R_L = 0.0           # Rezystancja szeregowa cewki [Ohm] - PSIM domyślnie 0
C = 0.2e-3          # Pojemność [F] - 200 µF (C12 w PSIM = 0.2m)
R_load = 50.0       # Rezystancja odbiornika [Ohm] (R33 w PSIM)

# ==============================================================================
# 2. PARAMETRY STEROWNIKA (dokładnie jak w PSIM C-block)
# ==============================================================================

u_ref = 240.0       # Napięcie referencyjne [V] - stałe (odczytane z wykresu PSIM)
wi = 0.5            # Waga błędu prądu
i_max = 20.0        # Limit prądu [A]

# Współczynniki filtru LP (z PSIM: 0.01, 0.01, 0.98)
alpha = 0.01
beta = 0.98

# ==============================================================================
# 3. PARAMETRY SYMULACJI
# ==============================================================================

t_end = 0.010       # Czas symulacji [s] - 10 ms (jak w PSIM)
dt = 1e-6           # Krok czasowy [s] - 1 µs

# Sterownik działa co 10 µs (Ts = 0.00001 w PSIM)
T_s_ctrl = 10e-6
ctrl_decimation = int(T_s_ctrl / dt)

# ==============================================================================
# 4. SYMULACJA - ALGORYTM IDENTYCZNY Z PSIM C-BLOCK
# ==============================================================================

t_vals = np.arange(0, t_end, dt)
n_steps = len(t_vals)

i_L_vals = np.zeros(n_steps)
v_C_vals = np.zeros(n_steps)
i_exp_vals = np.zeros(n_steps)
S_vals = np.zeros(n_steps)

# Warunki początkowe (Initial Capacitor Voltage = 100V w PSIM)
i_L = 0.0
v_C = 100.0
i_exp = 0.0
i_old = 0.0
i_exp_old = 0.0
s = 0.0

i_L_vals[0] = i_L
v_C_vals[0] = v_C

for k in range(1, n_steps):

    # --- STEROWNIK (co T_s_ctrl = 10µs, identycznie jak PSIM) ---
    if k % ctrl_decimation == 0:
        i_act = i_L
        uout_act = v_C

        # Linia 32 PSIM: error_u
        error_u = (1.0 + wi * i_act / u_ref) * u_ref - uout_act

        # Linia 35 PSIM: filtr LP dla i_exp (i_des)
        i_exp = alpha * i_act + alpha * i_old + beta * i_exp_old
        i_old = i_act
        i_exp_old = i_exp

        # Linia 41 PSIM: error_i (mnożnik 0 na i_exp!)
        error_i = 0 * i_exp - i_act

        # Linia 43 PSIM: funkcja przełączająca
        s = 1.0 if (error_u + wi * error_i) > 0 else 0.0

        # Linia 45-48 PSIM: zabezpieczenie nadprądowe
        if i_act > i_max:
            s = 0.0
        if i_act < -i_max:
            s = 1.0

    # --- FIZYKA UKŁADU (metoda Eulera, dt=1µs) ---
    if s == 1.0:
        # Tryb 1: Ładowanie cewki (switch ON)
        di_L = (V_in - i_L * R_L) / L
        dv_C = (-v_C / R_load) / C
    else:
        # Tryb 2: Boost (switch OFF)
        di_L = (V_in - i_L * R_L - v_C) / L
        dv_C = (i_L - v_C / R_load) / C

    i_L = i_L + di_L * dt
    v_C = v_C + dv_C * dt

    # Zapis
    i_L_vals[k] = i_L
    v_C_vals[k] = v_C
    i_exp_vals[k] = i_exp
    S_vals[k] = s

# ==============================================================================
# 5. WYKRESY (układ jak w PSIM)
# ==============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
t_ms = t_vals * 1000

# --- Napięcie wyjściowe ---
axes[0].plot(t_ms, v_C_vals, 'b', linewidth=1, label='v_C (Python)')
axes[0].axhline(y=u_ref, color='r', linestyle='--', linewidth=1.5, label=f'u_ref = {u_ref}V')
axes[0].set_ylabel('Napięcie [V]')
axes[0].set_title('Napięcie wyjściowe udc')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([50, 280])

# --- Prąd cewki ---
axes[1].plot(t_ms, i_L_vals, 'g', linewidth=0.8, label='i_L (Python)')
axes[1].plot(t_ms, i_exp_vals, 'r--', linewidth=1, label='i_exp (filtr LP)')
axes[1].axhline(y=i_max, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'i_max = ±{i_max}A')
axes[1].axhline(y=-i_max, color='gray', linestyle=':', linewidth=1, alpha=0.5)
axes[1].set_ylabel('Prąd [A]')
axes[1].set_title('Prąd cewki i_L i referencja i_exp')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# --- Stan przełącznika ---
axes[2].plot(t_ms, S_vals, 'k', linewidth=0.5)
axes[2].set_ylabel('PWM [-]')
axes[2].set_xlabel('Czas [ms]')
axes[2].set_title('Sygnał przełączający (1=ON/ładuj, 0=OFF/boost)')
axes[2].set_ylim([-0.1, 1.1])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle(
    f'Python Digital Twin — BB without load current estimation\n'
    f'V_in={V_in}V, L={L*1e3}mH, C={C*1e6:.0f}µF, R={R_load}Ω, u_ref={u_ref}V',
    fontsize=12, y=1.02
)
plt.savefig('wykres_bb_psim_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ==============================================================================
# 6. STATYSTYKI
# ==============================================================================

# Szukamy momentu ustabilizowania (v_C w zakresie ±5% od u_ref)
settled = np.where(np.abs(v_C_vals - u_ref) < u_ref * 0.05)[0]
if len(settled) > 0:
    t_settle = t_vals[settled[0]] * 1000
else:
    t_settle = float('nan')

# Stan ustalony - ostatnie 2ms
steady_start = int(0.008 / dt)
v_C_steady = v_C_vals[steady_start:]
i_L_steady = i_L_vals[steady_start:]

print("\n" + "=" * 60)
print("STATYSTYKI SYMULACJI (Python Digital Twin)")
print("=" * 60)
print(f"Parametry: L={L*1e3}mH, C={C*1e6:.0f}µF, R={R_load}Ω")
print(f"Napięcie referencyjne:         {u_ref:.1f} V")
print(f"Napięcie końcowe:              {v_C_vals[-1]:.2f} V")
print(f"Błąd stanu ustalonego:         {u_ref - np.mean(v_C_steady):.2f} V")
print(f"Tętnienia V (peak-to-peak):    {np.max(v_C_steady) - np.min(v_C_steady):.2f} V")
print(f"Prąd max:                      {np.max(i_L_vals):.2f} A")
print(f"Prąd min:                      {np.min(i_L_vals):.2f} A")
print(f"Czas stabilizacji (5%):        {t_settle:.2f} ms")
print(f"Średni duty cycle (koniec):    {np.mean(S_vals[steady_start:]):.2%}")
print("=" * 60)
