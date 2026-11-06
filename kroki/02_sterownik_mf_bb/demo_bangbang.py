"""
demo_bangbang.py — sterownik Model-Free Bang-Bang (MF-BB) w pętli zamkniętej (Krok 2).

Implementacja bezmodelowego sterownika Bang-Bang wg Artykułu 2 (Tatari, Bizhani, Iwański,
Power Electronics and Drives). Generuje sygnał odniesienia prądu cewki i_des przez filtr
LP 1. rzędu (alpha=0.01, beta=0.98) z mierzonego i_L, następnie steruje kluczem na
podstawie funkcji przełączającej:

    S = K_v · (u_ref - u_C) + w_i · (i_des - i_L)
    S > 0  →  klucz ON (gromadzimy energię w cewce)
    S ≤ 0  →  klucz OFF (energia trafia do kondensatora i obciążenia)

Scenariusz testowy: skok napięcia referencyjnego 200 V → 240 V w t=25 ms. Plik nie
porównuje z PSIM bezpośrednio — to wersja "robocza" sterownika; walidacja PSIM
realizowana w demo_bangbang_psim.py oraz porownanie_bb_psim.py.
"""
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PARAMETRY FIZYCZNE UKŁADU (wartości z PSIM - walidowane w Kroku 1)
# ==============================================================================

V_in = 100.0        # Napięcie wejściowe [V]
L = 0.75e-3         # Indukcyjność [H] - 0.75 mH
R_L = 0.05          # Rezystancja szeregowa cewki [Ohm]
C = 0.2e-3          # Pojemność [F] - 200 µF
R_load = 50.0       # Rezystancja odbiornika [Ohm]

# ==============================================================================
# 2. PARAMETRY STEROWNIKA BANG-BANG
# ==============================================================================

u_ref_initial = 200.0   # Początkowe napięcie referencyjne [V]
u_ref_final = 240.0     # Napięcie po skoku [V]
t_step = 0.025          # Czas skoku [s] - 25 ms

# Współczynniki filtru LP dla referencji prądu (z artykułu)
alpha = 0.01
beta = 0.98

# Waga błędu napięcia vs prądu
w_i = 0.5           # Zakres: 0.2-1.0

# Wzmocnienie błędu napięcia (skalowanie V → A)
K_v = 0.05          # [A/V]

# ==============================================================================
# 3. PARAMETRY SYMULACJI
# ==============================================================================

t_end = 0.05        # Czas symulacji [s] - 50 ms
dt = 1e-6           # Krok czasowy [s] - 1 µs

# Okres próbkowania sterownika
T_s_ctrl = 10e-6    # Sterownik działa co 10 µs (szybciej dla lepszej regulacji)
ctrl_decimation = int(T_s_ctrl / dt)

# ==============================================================================
# 4. SYMULACJA NUMERYCZNA
# ==============================================================================

t_vals = np.arange(0, t_end, dt)
n_steps = len(t_vals)

i_L_vals = np.zeros(n_steps)
v_C_vals = np.zeros(n_steps)
i_des_vals = np.zeros(n_steps)
S_vals = np.zeros(n_steps)
u_ref_vals = np.zeros(n_steps)  # Tablica napięcia referencyjnego (ze skokiem)

# Warunki początkowe
i_L = 0.0
v_C = 100.0
i_des = 0.0
i_L_prev = 0.0
S = 1  # Start od ładowania

i_L_vals[0] = i_L
v_C_vals[0] = v_C
i_des_vals[0] = i_des
S_vals[0] = S
u_ref_vals[0] = u_ref_initial

for k in range(1, n_steps):
    t = t_vals[k]

    # Napięcie referencyjne ze skokiem
    if t < t_step:
        u_ref = u_ref_initial
    else:
        u_ref = u_ref_final

    u_ref_vals[k] = u_ref

    # --- STEROWNIK BANG-BANG ---
    if k % ctrl_decimation == 0:
        # Aktualizacja referencji prądu przez filtr LP
        i_des_new = alpha * i_L + alpha * i_L_prev + beta * i_des
        i_L_prev = i_L
        i_des = i_des_new

        # === INTERPRETACJA ZE SKALOWANIEM ===
        # Błąd napięcia znormalizowany do zakresu prądowego
        # Gdy v_C < u_ref: chcemy zwiększyć napięcie → boost więcej
        # Skalowanie: 100V błędu ≈ kilka amperów prądu
        v_error_scaled = K_v * (u_ref - v_C)

        # Funkcja przełączająca:
        # S_func = v_error_scaled + w_i * (i_des - i_L)
        # - Gdy prąd niski (i_L < i_des): term dodatni → ładuj
        # - Gdy prąd wysoki (i_L > i_des): term ujemny → boost

        S_func = v_error_scaled + w_i * (i_des - i_L)

        # Przełączanie:
        if S_func > 0:
            S = 1   # Ładuj cewkę (prąd za niski LUB napięcie za niskie)
        else:
            S = 0   # Boost (prąd wysoki I napięcie wystarczające)

    # --- FIZYKA UKŁADU ---
    if S == 1:
        # Faza ładowania
        di_L = (V_in - i_L * R_L) / L
        dv_C = (-v_C / R_load) / C
    else:
        # Faza boost
        di_L = (V_in - i_L * R_L - v_C) / L
        dv_C = (i_L - v_C / R_load) / C

    # Metoda Eulera
    i_L = i_L + di_L * dt
    v_C = v_C + dv_C * dt

    # Zapis
    i_L_vals[k] = i_L
    v_C_vals[k] = v_C
    i_des_vals[k] = i_des
    S_vals[k] = S

# ==============================================================================
# 5. WYKRESY
# ==============================================================================

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
t_ms = t_vals * 1000

axes[0].plot(t_ms, v_C_vals, 'b', linewidth=1, label='v_C')
axes[0].plot(t_ms, u_ref_vals, 'r--', linewidth=2, label='u_ref')
axes[0].axvline(x=t_step*1000, color='gray', linestyle=':', linewidth=1, alpha=0.7)
axes[0].set_ylabel('Napięcie [V]')
axes[0].set_title(f'Napięcie wyjściowe (skok referencji: {u_ref_initial}→{u_ref_final}V przy t={t_step*1000:.0f}ms)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_ms, i_L_vals, 'g', linewidth=1, label='i_L')
axes[1].plot(t_ms, i_des_vals, 'r--', linewidth=1, label='i_des')
axes[1].set_ylabel('Prąd [A]')
axes[1].set_title('Prąd cewki i referencja')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t_ms, u_ref_vals - v_C_vals, 'm', linewidth=1)
axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[2].set_ylabel('Błąd [V]')
axes[2].set_title('Błąd napięcia (u_ref - v_C)')
axes[2].grid(True)

axes[3].plot(t_ms, S_vals, 'k', linewidth=0.5)
axes[3].set_ylabel('S [-]')
axes[3].set_xlabel('Czas [ms]')
axes[3].set_title('Stan przełącznika (1=ładuj, 0=boost)')
axes[3].set_ylim([-0.1, 1.1])
axes[3].grid(True)

plt.tight_layout()
plt.suptitle(f'Bang-Bang: V_in={V_in}V, C={C*1e6:.0f}µF, w_i={w_i}, K_v={K_v}', fontsize=12, y=1.02)
plt.show()

# ==============================================================================
# 6. STATYSTYKI
# ==============================================================================

steady_start = int(0.04 / dt)
v_C_steady = v_C_vals[steady_start:]

print("\n" + "="*60)
print("STATYSTYKI SYMULACJI")
print("="*60)
print(f"Napięcie referencyjne końcowe: {u_ref_final:.1f} V")
print(f"Napięcie końcowe:              {v_C_vals[-1]:.2f} V")
print(f"Błąd w stanie ustalonym:       {u_ref_final - np.mean(v_C_steady):.2f} V")
print(f"Tętnienia (peak-to-peak):      {np.max(v_C_steady) - np.min(v_C_steady):.2f} V")
print(f"Prąd końcowy cewki:            {i_L_vals[-1]:.2f} A")
print(f"Średni duty cycle (końcowy):   {np.mean(S_vals[steady_start:]):.2%}")
print("="*60)
