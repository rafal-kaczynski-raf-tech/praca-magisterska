"""
test_dt_sweep.py — sweep kroku całkowania dt przy stałym T_s_ctrl=10µs (Krok 2.5).

Diagnostyka wpływu dyskretyzacji metody Eulera na zgodność z PSIM. Skrypt uruchamia
symulację MF-BB dla różnych wartości dt (1µs, 5µs, 10µs, ...) zachowując stały okres
sterowania T_s_ctrl=10µs (separacja dyskretyzacji fizyki od dyskretyzacji sterownika)
i porównuje każdy wariant z psim_bb_wyniki.txt.

Cel: znaleźć dt dające najlepsze dopasowanie do PSIM przy akceptowalnym koszcie
obliczeniowym — ważne przed pętlą PSO, gdzie dt determinuje czas całej optymalizacji.
"""
import numpy as np

# Wczytanie PSIM
psim_data = np.loadtxt('psim_bb_wyniki.txt', skiprows=1)
psim_t = psim_data[:, 0]
psim_iL = psim_data[:, 1]
psim_udc = psim_data[:, 3]

# Parametry
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
T_s_ctrl = 10e-6
t_end = 0.010


def derivs(iL, vC, s_val):
    if s_val == 1.0:
        diL = (V_in - iL * R_L) / L
        dvC = (-vC / R_load) / C
    else:
        diL = (V_in - iL * R_L - vC) / L
        dvC = (iL - vC / R_load) / C
    return diL, dvC


def run_simulation(dt, integrator='trap'):
    ctrl_decimation = max(1, int(round(T_s_ctrl / dt)))
    n_steps = int(t_end / dt)
    t_vals = np.arange(n_steps) * dt

    i_L_vals = np.zeros(n_steps)
    v_C_vals = np.zeros(n_steps)

    i_L = 0.0
    v_C = 100.0
    i_exp = 0.0
    i_old = 0.0
    i_exp_old = 0.0
    s = 0.0

    i_L_vals[0] = i_L
    v_C_vals[0] = v_C

    for k in range(1, n_steps):
        # Fizyka
        if integrator == 'trap':
            d1 = derivs(i_L, v_C, s)
            iL_p = i_L + dt * d1[0]
            vC_p = v_C + dt * d1[1]
            d2 = derivs(iL_p, vC_p, s)
            i_L = i_L + dt/2 * (d1[0] + d2[0])
            v_C = v_C + dt/2 * (d1[1] + d2[1])
        elif integrator == 'euler':
            diL, dvC = derivs(i_L, v_C, s)
            i_L = i_L + dt * diL
            v_C = v_C + dt * dvC
        elif integrator == 'rk4':
            k1 = derivs(i_L, v_C, s)
            k2 = derivs(i_L + dt/2*k1[0], v_C + dt/2*k1[1], s)
            k3 = derivs(i_L + dt/2*k2[0], v_C + dt/2*k2[1], s)
            k4 = derivs(i_L + dt*k3[0], v_C + dt*k3[1], s)
            i_L = i_L + dt/6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
            v_C = v_C + dt/6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

        # Sterownik
        if k % ctrl_decimation == 0:
            error_u = (1.0 + wi * i_L / u_ref) * u_ref - v_C
            i_exp = alpha * i_L + alpha * i_old + beta * i_exp_old
            i_old = i_L
            i_exp_old = i_exp
            error_i = 0 * i_exp - i_L
            s = 1.0 if (error_u + wi * error_i) > 0 else 0.0
            if i_L > i_max: s = 0.0
            if i_L < -i_max: s = 1.0

        i_L_vals[k] = i_L
        v_C_vals[k] = v_C

    return t_vals, i_L_vals, v_C_vals


def metrics(t_vals, i_L_vals, v_C_vals, label):
    py_udc = np.interp(psim_t, t_vals, v_C_vals)
    py_iL = np.interp(psim_t, t_vals, i_L_vals)

    udc_err = psim_udc - py_udc
    iL_err = psim_iL - py_iL

    ss = psim_t > 0.006
    udc_mean_psim = np.mean(psim_udc[ss])
    udc_mean_py = np.mean(py_udc[ss])
    udc_pp_psim = np.ptp(psim_udc[ss])
    udc_pp_py = np.ptp(py_udc[ss])
    iL_mean_psim = np.mean(psim_iL[ss])
    iL_mean_py = np.mean(py_iL[ss])

    print(f"\n{label}")
    print(f"  MAE udc:        {np.mean(np.abs(udc_err)):6.3f} V")
    print(f"  MAE iL:         {np.mean(np.abs(iL_err)):6.3f} A")
    print(f"  Mean udc:       PSIM={udc_mean_psim:6.2f}V  Py={udc_mean_py:6.2f}V  Δ={udc_mean_py-udc_mean_psim:+5.2f}V")
    print(f"  PP udc:         PSIM={udc_pp_psim:6.2f}V  Py={udc_pp_py:6.2f}V  Δ={udc_pp_py-udc_pp_psim:+5.2f}V")
    print(f"  Mean iL:        PSIM={iL_mean_psim:6.2f}A  Py={iL_mean_py:6.2f}A  Δ={iL_mean_py-iL_mean_psim:+5.2f}A")


print("=" * 75)
print("TEST: różne dt + integratory (T_s_ctrl=10µs stały)")
print("=" * 75)

for dt_us in [10, 5, 2, 1, 0.5, 0.1]:
    dt = dt_us * 1e-6
    for integ in ['trap', 'euler', 'rk4']:
        t, iL, vC = run_simulation(dt, integ)
        metrics(t, iL, vC, f"dt={dt_us}µs, {integ.upper()}")
