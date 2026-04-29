"""
wykres_zoom_porownanie.py — wykres dwupanelowy: full-sim + zoom 0–2 ms (Krok 2.5).

Generuje wykres porównawczy Python vs PSIM w dwóch wariantach obok siebie:
  • lewa kolumna: pełna symulacja 0–10 ms (3 wiersze: napięcie, prąd, PWM),
  • prawa kolumna: zoom 0–2 ms — faza narastania napięcia, gdzie zgodność
    Python ↔ PSIM jest najwyższa (drobne przesunięcia fazowe pojawiają się
    dopiero ~0.4 ms i narastają w stanie ustalonym).

Kod symulacji Python skopiowany 1:1 z porownanie_bb_psim.py (te same parametry,
ZOH 1-sample delay, dt=10µs, Euler) — gwarantuje że tu zobaczysz dokładnie ten
sam stan układu co na wykresie finalnym, tylko z dodatkowym zoom-em.

Wynik: wykres_zoom_porownanie.png
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
psim_iL = psim_data[:, 1]
psim_udc = psim_data[:, 3]
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

# ==============================================================================
# 3. SYMULACJA PYTHON (kopia algorytmu z porownanie_bb_psim.py)
# ==============================================================================

t_vals = np.arange(0, t_end, dt)
n_steps = len(t_vals)

i_L_vals = np.zeros(n_steps)
v_C_vals = np.zeros(n_steps)
S_vals = np.zeros(n_steps)

i_L = 0.0
v_C = 100.0
i_exp = 0.0
i_old = 0.0
i_exp_old = 0.0
s = 0.0

i_L_prev_sample = 0.0
v_C_prev_sample = 100.0

i_L_vals[0] = i_L
v_C_vals[0] = v_C


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

    i_L_vals[k] = i_L
    v_C_vals[k] = v_C
    diL, dvC = derivs(i_L, v_C, s)
    i_L = i_L + diL * dt
    v_C = v_C + dvC * dt
    S_vals[k] = s

# ==============================================================================
# 4. WYKRES DWUPANELOWY (3 wiersze x 2 kolumny)
# ==============================================================================

t_ms = t_vals * 1000
psim_t_ms = psim_t * 1000

fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col')

# --- Konfiguracja kolumn ---
column_configs = [
    {'col': 0, 'xlim': (0, 10),  'title_suffix': 'pełna symulacja (0–10 ms)'},
    {'col': 1, 'xlim': (0, 2),   'title_suffix': 'zoom 0–2 ms (faza narastania)'},
]

for cfg in column_configs:
    c = cfg['col']

    # Wiersz 0: napięcie wyjściowe
    axes[0, c].plot(psim_t_ms, psim_udc, 'r', linewidth=1.5, label='PSIM (udc_BB1)', alpha=0.85)
    axes[0, c].plot(t_ms, v_C_vals, 'b--', linewidth=1.0, label='Python (v_C)', alpha=0.85)
    axes[0, c].axhline(y=u_ref, color='gray', linestyle=':', linewidth=1, alpha=0.5,
                       label=f'u_ref = {u_ref} V')
    axes[0, c].set_ylabel('Napięcie [V]')
    axes[0, c].set_title(f'Napięcie wyjściowe — {cfg["title_suffix"]}')
    axes[0, c].legend(loc='lower right', fontsize=9)
    axes[0, c].grid(True, alpha=0.3)
    axes[0, c].set_xlim(cfg['xlim'])

    # Wiersz 1: prąd cewki
    axes[1, c].plot(psim_t_ms, psim_iL, 'r', linewidth=1.0, label='PSIM I(L7)', alpha=0.85)
    axes[1, c].plot(t_ms, i_L_vals, 'b--', linewidth=0.8, label='Python i_L', alpha=0.85)
    axes[1, c].set_ylabel('Prąd [A]')
    axes[1, c].set_title(f'Prąd cewki — {cfg["title_suffix"]}')
    axes[1, c].legend(loc='upper right', fontsize=9)
    axes[1, c].grid(True, alpha=0.3)
    axes[1, c].set_xlim(cfg['xlim'])

    # Wiersz 2: PWM
    axes[2, c].plot(psim_t_ms, psim_pwm, 'r', linewidth=0.8, label='PSIM pwmBB1', alpha=0.75)
    axes[2, c].plot(t_ms, S_vals, 'b--', linewidth=0.5, label='Python S', alpha=0.75)
    axes[2, c].set_ylabel('PWM [-]')
    axes[2, c].set_xlabel('Czas [ms]')
    axes[2, c].set_title(f'Sygnał przełączający — {cfg["title_suffix"]}')
    axes[2, c].legend(loc='upper right', fontsize=9)
    axes[2, c].set_ylim([-0.1, 1.1])
    axes[2, c].grid(True, alpha=0.3)
    axes[2, c].set_xlim(cfg['xlim'])

plt.tight_layout()
plt.suptitle(
    'Walidacja Digital Twin: Python vs PSIM — porównanie pełna symulacja vs zoom\n'
    f'BB without load current estimation — L={L*1e3}mH, C={C*1e6:.0f}µF, R={R_load}Ω, u_ref={u_ref}V',
    fontsize=13, fontweight='bold', y=1.02
)
plt.savefig('wykres_zoom_porownanie.png', dpi=150, bbox_inches='tight')
print("Zapisano: wykres_zoom_porownanie.png")
