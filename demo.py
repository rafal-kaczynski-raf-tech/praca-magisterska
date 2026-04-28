import numpy as np          # NumPy - biblioteka do obliczeń matematycznych, tworzy tablice liczb
import matplotlib.pyplot as plt  # Matplotlib - biblioteka do rysowania wykresów
import pandas as pd         # Pandas - biblioteka do wczytywania plików CSV (tabelek z danymi)
import os                   # os - moduł do sprawdzania czy plik istnieje na dysku

# ==============================================================================
# 1. PARAMETRY FIZYCZNE UKŁADU (wartości odczytane z modelu PSIM)
# ==============================================================================

# Źródło napięcia
V_in = 100.0        # Napięcie wejściowe [V] - stałe napięcie zasilania 100V (DC Offset w PSIM)

# Cewka (induktor L7)
L = 0.75e-3         # Indukcyjność [H] - 0.75 mH, cewka gromadzi energię w polu magnetycznym
R_L = 0.05          # Rezystancja szeregowa cewki [Ohm] - mały opór drutu, powoduje straty ciepła

# Kondensator (C12)
C = 0.2e-3          # Pojemność [F] - 0.2 mF (200 µF), kondensator gromadzi energię w polu elektrycznym

# Obciążenie
R_load = 50.0       # Rezystancja odbiornika [Ohm] - 50 Ohm, pobiera prąd z kondensatora

# Parametry sterowania PWM (z Gating Block w PSIM)
f_sw = 2000.0       # Częstotliwość przełączania [Hz] - tranzystory przełączają się 2000 razy na sekundę
T_sw = 1.0 / f_sw   # Okres przełączania [s] - czas jednego pełnego cyklu = 0.5 ms
D = 0.5             # Współczynnik wypełnienia [-] - 50%, czyli tranzystor jest włączony przez połowę cyklu

# Parametry symulacji
t_end = 0.01        # Czas końcowy symulacji [s] - 10 ms (tyle samo co w PSIM)
dt = 1e-6           # Krok czasowy [s] - 1 µs, im mniejszy tym dokładniejsza symulacja

# ==============================================================================
# 2. SYMULACJA NUMERYCZNA (obliczenia krok po kroku metodą Eulera)
# ==============================================================================

# Tworzymy tablice do przechowywania wyników
t_vals = np.arange(0, t_end, dt)    # Oś czasu: tablica wartości od 0 do 10 ms co 1 µs (10 000 punktów)
i_L_vals = np.zeros_like(t_vals)    # Tablica na prąd cewki - na razie wypełniona zerami
v_C_vals = np.zeros_like(t_vals)    # Tablica na napięcie kondensatora - na razie wypełniona zerami

# Warunki początkowe (takie same jak w PSIM)
i_L = 0.0           # Prąd początkowy cewki [A] - cewka startuje "pusta", bez prądu
v_C = 100.0         # Napięcie początkowe kondensatora [V] - w PSIM ustawione jako "Initial Capacitor Voltage"

# Zapisujemy warunki początkowe do tablic
i_L_vals[0] = i_L
v_C_vals[0] = v_C

# Główna pętla symulacji - powtarza się dla każdego kroku czasowego
for k in range(1, len(t_vals)):
    t = t_vals[k]   # Aktualny czas symulacji

    # Sprawdzamy w której fazie cyklu PWM jesteśmy
    t_in_period = t % T_sw  # Czas wewnątrz obecnego okresu przełączania (0 do 0.5 ms)

    # Określamy stan tranzystora (S=1: dolny włączony, S=0: górny włączony)
    if t_in_period < D * T_sw:
        S = 1   # Pierwsza połowa cyklu (0-180°): dolny tranzystor ON - cewka ładuje się z zasilacza
    else:
        S = 0   # Druga połowa cyklu (180-360°): górny tranzystor ON - cewka oddaje energię do kondensatora

    # Obliczamy zmiany prądu i napięcia zgodnie z prawami fizyki (równania różniczkowe)
    if S == 1:
        # FAZA ŁADOWANIA CEWKI: cewka połączona z zasilaczem, odcięta od kondensatora
        # Napięcie na cewce: V_L = V_in - i_L * R_L
        # Prawo indukcji: di/dt = V_L / L
        di_L = (V_in - i_L * R_L) / L
        # Kondensator rozładowuje się przez obciążenie (nie dostaje prądu z cewki)
        dv_C = (-v_C / R_load) / C
    else:
        # FAZA BOOST: cewka połączona z kondensatorem, oddaje zgromadzoną energię
        # Napięcie na cewce: V_L = V_in - i_L * R_L - v_C (musi "pokonać" napięcie kondensatora)
        di_L = (V_in - i_L * R_L - v_C) / L
        # Kondensator ładuje się prądem z cewki, jednocześnie oddając prąd do obciążenia
        dv_C = (i_L - v_C / R_load) / C

    # Metoda Eulera: nowa wartość = stara wartość + zmiana * krok czasowy
    i_L = i_L + di_L * dt   # Aktualizujemy prąd cewki
    v_C = v_C + dv_C * dt   # Aktualizujemy napięcie kondensatora

    # Uwaga: To jest układ synchroniczny z dwoma tranzystorami (nie z diodą),
    # dlatego prąd może płynąć w obu kierunkach (także ujemny).

    # Zapisujemy wyniki do tablic
    i_L_vals[k] = i_L
    v_C_vals[k] = v_C

# ==============================================================================
# 3. WCZYTANIE DANYCH REFERENCYJNYCH Z PSIM
# ==============================================================================

plik_csv = 'psim_wyniki.csv'    # Nazwa pliku z wyeksportowanymi danymi z PSIM
dane_psim_dostepne = False      # Flaga: czy udało się wczytać dane z PSIM

if os.path.exists(plik_csv):
    try:
        # Wczytujemy plik CSV do tabeli (DataFrame)
        psim_df = pd.read_csv(plik_csv)
        psim_df.columns = psim_df.columns.str.strip()  # Usuwamy ewentualne spacje z nazw kolumn

        # Wyciągamy poszczególne kolumny z danymi
        t_psim = psim_df['Time']     # Czas [s]
        v_psim = psim_df['V2']       # Napięcie z sondy V2 [V]
        i_psim = psim_df['I(L7)']    # Prąd z sondy na cewce L7 [A]

        dane_psim_dostepne = True
        print("Pomyślnie wczytano dane z PSIM!")

    except Exception as e:
        print(f"Błąd podczas czytania pliku CSV: {e}")
        print("Sprawdź czy plik ma kolumny: Time, V2, I(L7)")
else:
    print(f"Nie znaleziono pliku '{plik_csv}' w folderze ze skryptem.")
    print("Wykresy pokażą tylko wyniki z Pythona (bez porównania z PSIM).")

# ==============================================================================
# 4. RYSOWANIE WYKRESÓW
# ==============================================================================

# Tworzymy okno z dwoma wykresami (góra: napięcie, dół: prąd)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- WYKRES GÓRNY: Napięcie na kondensatorze ---
ax1.plot(t_vals, v_C_vals, 'r', linewidth=3, label='Python (Model Matematyczny)')
if dane_psim_dostepne:
    ax1.plot(t_psim, v_psim, 'k--', linewidth=2, label='PSIM (Sonda V2)')
ax1.set_ylabel('Napięcie [V]')
ax1.set_title('Porównanie napięcia na kondensatorze (v_C)')
ax1.grid(True)      # Siatka ułatwia odczytywanie wartości
ax1.legend()        # Legenda wyjaśnia co oznaczają kolory linii

# --- WYKRES DOLNY: Prąd cewki ---
ax2.plot(t_vals, i_L_vals, 'g', linewidth=3, label='Python (Model Matematyczny)')
if dane_psim_dostepne:
    ax2.plot(t_psim, i_psim, 'k--', linewidth=2, label='PSIM (Sonda I(L7))')
ax2.set_ylabel('Prąd [A]')
ax2.set_xlabel('Czas [s]')
ax2.set_title('Porównanie prądu cewki (i_L)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()  # Automatyczne dostosowanie marginesów
plt.show()          # Wyświetlenie okna z wykresami
