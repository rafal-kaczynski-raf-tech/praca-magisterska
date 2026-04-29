"""
test_first_samples.py — analiza pierwszych 12 próbek PSIM vs Python (Krok 2.5).

Diagnoza rozbieżności w stanie przejściowym sterownika MF-BB. Porównuje próbka po
próbce wyniki PSIM (psim_bb_wyniki.txt, pierwsze 12 wierszy) i Python dla różnych
konfiguracji eksperymentalnych:

  - record_when:    'before_physics' vs 'after_physics' (kolejność zapis/symulacja)
  - initial_s:      0 (klucz OFF) vs 1 (klucz ON) na starcie

Cel: ustalić w jakim wariancie konwencji zapisu i stanu początkowego klucza
symulacja Python dokładnie odtwarza pierwsze próbki referencyjne z PSIM
(walidacja drobnych szczegółów implementacyjnych C-blocku PSIM).
"""
import numpy as np

psim_data = np.loadtxt('psim_bb_wyniki.txt', skiprows=1)
psim_t = psim_data[:, 0]
psim_iL = psim_data[:, 1]
psim_udc = psim_data[:, 3]
psim_pwm = psim_data[:, 7]

# Parametry
V_in = 100.0
L = 0.75e-3
R_L = 0.05
C = 0.2e-3
R_load = 50.0
u_ref = 240.0
wi = 0.5
i_max = 20.0
dt = 10e-6
T_s_ctrl = 10e-6
ctrl_decimation = 1


def run_sim(initial_s, record_when='before_physics'):
    """record_when: 'before_physics' or 'after_physics'"""
    n_steps = 12
    t_vals = np.arange(n_steps) * dt
    i_L_vals = np.zeros(n_steps)
    v_C_vals = np.zeros(n_steps)
    s_vals = np.zeros(n_steps)

    i_L = 0.0
    v_C = 100.0
    s = float(initial_s)

    i_L_vals[0] = i_L
    v_C_vals[0] = v_C
    s_vals[0] = s

    for k in range(1, n_steps):
        if record_when == 'before_physics':
            # Sterownik z bieżącym stanem
            error_u = (1.0 + wi * i_L / u_ref) * u_ref - v_C
            error_i = -i_L
            s = 1.0 if (error_u + wi * error_i) > 0 else 0.0
            if i_L > i_max: s = 0.0
            if i_L < -i_max: s = 1.0

            i_L_vals[k] = i_L
            v_C_vals[k] = v_C
            s_vals[k] = s

            if s == 1.0:
                di_L = (V_in - i_L * R_L) / L
                dv_C = (-v_C / R_load) / C
            else:
                di_L = (V_in - i_L * R_L - v_C) / L
                dv_C = (i_L - v_C / R_load) / C
            i_L = i_L + di_L * dt
            v_C = v_C + dv_C * dt

        elif record_when == 'after_physics':
            if s == 1.0:
                di_L = (V_in - i_L * R_L) / L
                dv_C = (-v_C / R_load) / C
            else:
                di_L = (V_in - i_L * R_L - v_C) / L
                dv_C = (i_L - v_C / R_load) / C
            i_L = i_L + di_L * dt
            v_C = v_C + dv_C * dt

            error_u = (1.0 + wi * i_L / u_ref) * u_ref - v_C
            error_i = -i_L
            s = 1.0 if (error_u + wi * error_i) > 0 else 0.0
            if i_L > i_max: s = 0.0
            if i_L < -i_max: s = 1.0

            i_L_vals[k] = i_L
            v_C_vals[k] = v_C
            s_vals[k] = s

    return t_vals, i_L_vals, v_C_vals, s_vals


print(f"\n{'t [µs]':<8} {'PSIM I(L7)':<12} {'PSIM udc':<10} {'PSIM pwm':<10}")
print("-" * 50)
for i in range(11):
    print(f"{psim_t[i]*1e6:<8.0f} {psim_iL[i]:<12.6f} {psim_udc[i]:<10.4f} {psim_pwm[i]:<10.0f}")

print("\n=== Python: skip first physics (rec BEFORE) ===")
print(f"{'t [µs]':<8} {'i_L':<12} {'v_C':<10} {'s':<10}")
print("-" * 50)
t, iL, vC, s = run_sim(initial_s=0, record_when='before_physics')
for i in range(11):
    print(f"{t[i]*1e6:<8.0f} {iL[i]:<12.6f} {vC[i]:<10.4f} {s[i]:<10.0f}")

print("\n=== Python: physics first, then record (rec AFTER) ===")
print(f"{'t [µs]':<8} {'i_L':<12} {'v_C':<10} {'s':<10}")
print("-" * 50)
t, iL, vC, s = run_sim(initial_s=0, record_when='after_physics')
for i in range(11):
    print(f"{t[i]*1e6:<8.0f} {iL[i]:<12.6f} {vC[i]:<10.4f} {s[i]:<10.0f}")
