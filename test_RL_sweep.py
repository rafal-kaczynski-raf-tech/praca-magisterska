"""
test_RL_sweep.py — sweep rezystancji szeregowej cewki R_L (Krok 2.5, kalibracja).

Diagnostyka rozbieżności Python vs PSIM w sterowaniu MF-BB. Hipoteza: PSIM ma
nieoczywiste straty omowe pomimo nominalnego R_L=0. Skrypt uruchamia symulację
MF-BB dla wachlarza wartości R_L i porównuje każdy wariant z psim_bb_wyniki.txt
celem znalezienia R_L minimalizującego MAE napięcia wyjściowego.

Użyteczny jako narzędzie zatwierdzania parametrów pasożytniczych przed
rozpoczęciem strojenia PSO.
"""
import numpy as np

psim_data = np.loadtxt('psim_bb_wyniki.txt', skiprows=1)
psim_t = psim_data[:, 0]
psim_iL = psim_data[:, 1]
psim_udc = psim_data[:, 3]

# Parametry
V_in = 100.0
L = 0.75e-3
C = 0.2e-3
R_load = 50.0
u_ref = 240.0
wi = 0.5
i_max = 20.0
dt = 1e-6
T_s_ctrl = 10e-6
ctrl_decimation = int(T_s_ctrl / dt)
t_end = 0.010


def run_sim(R_L_test):
    n_steps = int(t_end / dt)
    i_L_vals = np.zeros(n_steps)
    v_C_vals = np.zeros(n_steps)

    i_L = 0.0
    v_C = 100.0
    s = 0.0

    for k in range(1, n_steps):
        # Euler
        if s == 1.0:
            di_L = (V_in - i_L * R_L_test) / L
            dv_C = (-v_C / R_load) / C
        else:
            di_L = (V_in - i_L * R_L_test - v_C) / L
            dv_C = (i_L - v_C / R_load) / C

        i_L = i_L + di_L * dt
        v_C = v_C + dv_C * dt

        if k % ctrl_decimation == 0:
            error_u = (1.0 + wi * i_L / u_ref) * u_ref - v_C
            error_i = -i_L
            s = 1.0 if (error_u + wi * error_i) > 0 else 0.0
            if i_L > i_max: s = 0.0
            if i_L < -i_max: s = 1.0

        i_L_vals[k] = i_L
        v_C_vals[k] = v_C

    return i_L_vals, v_C_vals


t_vals = np.arange(int(t_end / dt)) * dt
ss_mask_psim = psim_t > 0.006

print("=" * 78)
print("TEST: różne R_L (rezystancja szeregowa cewki)")
print("=" * 78)
print(f"{'R_L [Ω]':<12} {'Mean udc Δ':<15} {'PP udc Δ':<15} {'Mean iL Δ':<15}")
print("-" * 78)

for R_L_test in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    iL, vC = run_sim(R_L_test)
    py_udc = np.interp(psim_t, t_vals, vC)
    py_iL = np.interp(psim_t, t_vals, iL)

    udc_d = np.mean(py_udc[ss_mask_psim]) - np.mean(psim_udc[ss_mask_psim])
    pp_d = np.ptp(py_udc[ss_mask_psim]) - np.ptp(psim_udc[ss_mask_psim])
    iL_d = np.mean(py_iL[ss_mask_psim]) - np.mean(psim_iL[ss_mask_psim])

    print(f"{R_L_test:<12.3f} {udc_d:+.3f} V        {pp_d:+.3f} V        {iL_d:+.3f} A")
