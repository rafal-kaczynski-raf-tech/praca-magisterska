import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETRY FIZYCZNE (MÓJ WIRTUALNY SPRZĘT)
# ==========================================
# Traktuję to jako stałe konfiguracyjne mojego układu.

V_in = 100.0         # Moje główne zasilanie. Traktuję to jak idealną baterię 100V.
L = 0.75e-3          # Cewka (0.75 mH). Działa jak koło zamachowe dla prądu - nie pozwala mu skakać nagle.
R_L = 0.05           # Opór drutu na cewce. Nawet miedź stawia opór, więc tu ucieka mi trochę energii w ciepło.
C = 200e-6           # Kondensator (200 uF). Mój bufor, który gromadzi ładunek, żeby wygładzić napięcie na wyjściu.
V_c_init = 100.0     # Napięcie startowe. Mój model nie startuje z pustym kondensatorem, zakładam na wejściu od razu 100V, tak jak w PSIM.

# Obciążenie, czyli to, co zżera mój prąd na samym końcu układu.
R_load1 = 50.0       # Główny opornik.
R_load2 = 1000.0     # Ten mały, dodatkowy opornik ze schematu.
# Rezystancja zastępcza (R_eq). Prąd ma dwie drogi ucieczki, więc liczę opór ze wzoru na połączenie równoległe.
R_eq = (R_load1 * R_load2) / (R_load1 + R_load2) 

# ==========================================
# 2. USTAWIENIA ZEGARA I PAMIĘCI
# ==========================================
dt = 1e-05           # Krok czasu (10 us). To moje "delta time". Co tyle czasu zatrzymuję świat i przeliczam fizykę.
t_total = 0.01       # Czas symulacji (10 ms). Tyle czasu wirtualnego świata chcę wygenerować.
N_steps = int(t_total / dt) # Liczba klatek (1000). Dzielę czas całkowity przez krok.

# Przygotowuję puste tablice na logi z symulacji.
time = np.linspace(0, t_total, N_steps) # Oś czasu
i_L = np.zeros(N_steps) # Historia prądu cewki
v_C = np.zeros(N_steps) # Historia napięcia kondensatora

# Stan na starcie (Klatka nr 0).
v_C[0] = V_c_init    # Kondensator ma od razu 100 V.
i_L[0] = 0.0         # Prąd na starcie wynosi zero.

# ==========================================
# 3. GENERATOR PWM (MÓJ "GŁUPI" GATING BLOCK)
# ==========================================
# Zastępuję skomplikowany sterownik profesora prostym metronomem.
f_sw = 2000.0        # Tranzystor będzie klikał 2000 razy na sekundę (2 kHz).
T_sw = 1.0 / f_sw    # Długość jednego cyklu wynosi 0.5 ms.
Duty = 0.5           # Wypełnienie 50%. Przez pół cyklu tranzystor przewodzi, przez resztę nie.

print("Odpalam silnik fizyczny...")

# ==========================================
# 4. GŁÓWNA PĘTLA SYMULACYJNA
# ==========================================
# Przeliczam klatkę 'k' i na jej podstawie przewiduję przyszłość w klatce 'k+1'.
for k in range(N_steps - 1):
    
    # Gdzie dokładnie jestem w obecnym cyklu PWM?
    t_in_period = time[k] % T_sw
    
    # Jeśli mój czas jest w pierwszej połowie cyklu, włączam tranzystor (S = 1). Inaczej wyłączam (S = 0).
    S = 1 if t_in_period < (Duty * T_sw) else 0
    
    # Równania różniczkowe - czyli liczenie "prędkości" zmian dla obecnej klatki.
    if S == 1:
        # STAN 1: TRANZYSTOR WŁĄCZONY. 
        # Cewka jest ładowana z baterii. Kondensator jest odcięty i tylko oddaje prąd do obciążenia.
        di_dt = (V_in - i_L[k] * R_L) / L
        dv_dt = -v_C[k] / (R_eq * C)
    else:
        # STAN 2: TRANZYSTOR WYŁĄCZONY.
        # Cewka oddaje zgromadzoną energię, pchając prąd prosto do kondensatora i obciążenia.
        di_dt = (V_in - v_C[k] - i_L[k] * R_L) / L
        dv_dt = (i_L[k] - v_C[k] / R_eq) / C
        
    # METODA EULERA (Krok w przyszłość)
    # Prosta zasada: Nowy stan = Stary stan + (Prędkość zmian * delta czasu).
    i_L[k+1] = i_L[k] + di_dt * dt
    v_C[k+1] = v_C[k] + dv_dt * dt

print("Koniec obliczeń. Generuję wykresy.")

# ==========================================
# 5. WYKRESY
# ==========================================
plt.figure(figsize=(10, 6))

# Wykres napięcia
plt.subplot(2, 1, 1)
plt.plot(time * 1000, v_C, label='Napięcie kondensatora ($V_C$)', color='#0052cc', linewidth=1.5)
plt.title('Bliźniak Cyfrowy: Otwarta Pętla (Fizyka bez sterownika)')
plt.ylabel('Napięcie [V]')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# Wykres prądu
plt.subplot(2, 1, 2)
plt.plot(time * 1000, i_L, label='Prąd cewki ($I_L$)', color='#d63031', linewidth=1.5)
plt.xlabel('Czas [ms]')
plt.ylabel('Prąd [A]')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()