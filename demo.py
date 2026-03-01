import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETRY FIZYCZNE (PROSTO Z PSIM)
# ==========================================
V_in = 100.0         # Napięcie wejściowe [V]
L = 0.75e-3          # Cewka 0.75 mH [H]
R_L = 0.05           # Rezystancja drutu cewki [Ohm]
C = 200e-6           # Kondensator 0.2 mF [F]
V_c_init = 100.0     # Napięcie początkowe kondensatora [V]

R_load1 = 50.0       # Główne obciążenie [Ohm]
R_load2 = 1000.0     # Dodatkowy rezystor [Ohm]
# Rezystancja zastępcza (równoległa)
R_eq = (R_load1 * R_load2) / (R_load1 + R_load2) 

# Parametry kluczy
R_on = 1e-05         # R_switch_on [Ohm]
R_off = 1e+07        # R_switch_off [Ohm]

# ==========================================
# 2. USTAWIENIA ZEGARA (SIMULATION CONTROL)
# ==========================================
dt = 1e-05           # Time step
t_total = 0.01       # Total Time
N_steps = int(t_total / dt)
time = np.linspace(0, t_total, N_steps)

# Inicjalizacja tablic na wyniki
i_L = np.zeros(N_steps)
v_C = np.zeros(N_steps)

# Warunki początkowe
v_C[0] = V_c_init 
i_L[0] = 0.0         # Zakładamy start od zerowego prądu cewki

print("Inicjalizacja środowiska zakończona sukcesem!")
print(f"Liczba kroków symulacji: {N_steps}")
# Tu docelowo wstawimy pętlę rozwiązującą równania różniczkowe...