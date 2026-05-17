"""
Replay-test: walidacja 1:1 modelu Python z PSIM w "oknach" zapisanych w CSV.

Dane PSIM to 12 segmentów po 40 próbek (20 µs każdy). Dla każdego segmentu
inicjalizujemy symulator Pythona dokładnie tym, co PSIM ma w pierwszej próbce
(i_L, v_C, s, pamięć sterownika, stan filtra LPF) i puszczamy 20 µs do przodu.
Jeśli model jest matematycznie identyczny z PSIM — krzywe pokryją się 1:1.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import demo_bangbang_estim as sim

CSV_PATH = "psim_bb_estim_wyniki.csv"
DT       = 0.5e-6
TS_CTRL  = 10e-6
N_PER_CTRL = int(round(TS_CTRL / DT))   # 20


def find_segments(df, dt_max=1e-6):
    t = df["Time"].values
    dt = np.diff(t)
    gaps = np.where(dt > dt_max)[0]
    starts = np.concatenate([[0], gaps + 1])
    ends   = np.concatenate([gaps + 1, [len(t)]])
    return list(zip(starts, ends))


def replay_segment(df_seg):
    """Symuluj Python startując z pierwszej próbki segmentu PSIM."""
    n = len(df_seg)
    # Stan początkowy z PSIM:
    iL = df_seg["I_L6_"].iloc[0]                 # rzeczywisty prąd cewki (BB)
    vC = df_seg["udc_BB"].iloc[0]                # napięcie wyjściowe
    s  = int(df_seg["pwmBB"].iloc[0])            # stan PWM

    # Stan sterownika na początku segmentu — wartości "static" z C-block:
    # PSIM zapisał iout_est_BB i iout_est_filt_BB w pierwszej próbce — to wartości
    # PO ostatnim wywołaniu sterownika sprzed startu segmentu.
    iout_est_init  = df_seg["iout_est_BB"].iloc[0]
    iout_filt_init = df_seg["iout_est_filt_BB"].iloc[0]

    # Stan filtra LPF: zakładamy że przed segmentem był ustalony (x1=x2=iout_est, y1=y2=iout_filt)
    x1 = iout_est_init;  x2 = iout_est_init
    y1 = iout_filt_init; y2 = iout_filt_init

    # Pamięć sterownika
    # i_old = i_act z poprzedniego wywołania sterownika.
    # Najlepsza dostępna estymata: iL_BB jest "delayed" wartością widzianą przez sterownik
    # więc iL_BB[0] ≈ i_act z poprzedniego cyklu, a iL_BB[?] z poprzedniego cyklu
    # nie ma — przyjmujemy i_old = iL_BB[0] (sterownik nie odpali w segmencie i tak,
    # bo segment ma 20µs = 1 cykl, a transition pwmBB jest na pozycji 19 modulo 20).
    i_old    = df_seg["iL_BB"].iloc[0]
    s_old    = s
    uout_old = vC
    iout_filt = iout_filt_init

    # Bufory na wynik
    t_arr  = np.zeros(n)
    iL_arr = np.zeros(n)
    vC_arr = np.zeros(n)
    s_arr  = np.zeros(n, dtype=np.int8)
    t_arr[0]=df_seg["Time"].iloc[0]; iL_arr[0]=iL; vC_arr[0]=vC; s_arr[0]=s

    # ZOH delay - nie wiemy dokładnie kiedy w segmencie wypadnie krok sterownika.
    # Z analizy: pwmBB zmienia się na pozycji 19 w segmencie (= 19 * 0.5µs = 9.5µs)
    # więc krok sterownika wypada między próbką 19 a 20. Liczymy od pozycji 19.
    # Przed tym krokiem sterownik nic nie robi — tylko fizyka.

    # Ale dla bezpieczeństwa: znajdźmy wszystkie próbki gdzie możemy wywołać sterownik.
    # Sterownik liczy się gdy "minęła pełna 10us". Bezpieczne kryterium: t mod Ts ≈ 0.
    t_start_seg = df_seg["Time"].iloc[0]

    # Wartości próbkowane (ZOH) — to co sterownik widział przy ostatnim wywołaniu
    i_L_prev_sample = df_seg["iL_BB"].iloc[0]    # to JEST opóźniona wartość!
    v_C_prev_sample = df_seg["udc_BB"].iloc[0]   # tu nie mamy "delayed" w PSIM, użyjmy bieżącej

    for k in range(1, n):
        t_now = df_seg["Time"].iloc[k]
        # Czy w tym kroku wypada nowy moment próbkowania sterownika?
        # Krok sterownika zawsze co 10µs. Sprawdźmy odległość od najbliższej wielokrotności.
        prev_t = t_arr[k-1]
        # Numer cyklu sterownika dla obu chwil:
        n_prev = int(np.floor(prev_t / TS_CTRL + 1e-9))
        n_curr = int(np.floor(t_now / TS_CTRL + 1e-9))
        if n_curr > n_prev:
            # nadszedł nowy krok sterownika
            i_act    = i_L_prev_sample
            uout_act = v_C_prev_sample

            iout_est = (1.0 - s_old) * 0.5 * (i_act + i_old) - sim.C * (uout_act - uout_old) / TS_CTRL
            iout_filt = sim.B0*iout_est + sim.B1*x1 + sim.B2*x2 - sim.A1*y1 - sim.A2*y2
            x2=x1; x1=iout_est; y2=y1; y1=iout_filt

            i_des = iout_filt * sim.u_ref / sim.V_in
            error_u = sim.u_ref - uout_act
            error_i = i_des - i_act
            s_new = 1 if (error_u + sim.wi * error_i) > 0 else 0
            if i_act >  sim.i_max: s_new = 0
            if i_act < -sim.i_max: s_new = 1

            s_old = s
            uout_old = uout_act
            i_old = i_act
            s = s_new

            i_L_prev_sample = iL
            v_C_prev_sample = vC

        # Fizyka — krok 0.5µs
        v_L = sim.V_in - (1.0 - s) * vC - sim.R_L * iL
        i_C = (1.0 - s) * iL - vC / sim.R_load
        iL = iL + v_L / sim.L * DT
        vC = vC + i_C / sim.C * DT

        t_arr[k]=t_now; iL_arr[k]=iL; vC_arr[k]=vC; s_arr[k]=s

    return t_arr, iL_arr, vC_arr, s_arr


def main():
    df = pd.read_csv(CSV_PATH).sort_values("Time").reset_index(drop=True)
    segs = find_segments(df)
    print(f"Znaleziono {len(segs)} segmentów ciągłych.")

    n_show = min(6, len(segs))
    fig, axes = plt.subplots(n_show, 2, figsize=(14, 2.4*n_show), sharex="row")
    if n_show == 1: axes = axes.reshape(1, 2)

    diffs_iL = []
    diffs_vC = []

    for i, (s_idx, e_idx) in enumerate(segs[:n_show]):
        seg = df.iloc[s_idx:e_idx].reset_index(drop=True)
        t_psim = seg["Time"].values
        iL_psim = seg["I_L6_"].values
        vC_psim = seg["udc_BB"].values

        t_py, iL_py, vC_py, s_py = replay_segment(seg)

        ax_i = axes[i, 0]
        ax_v = axes[i, 1]
        ax_i.plot((t_psim - t_psim[0])*1e6, iL_psim, "o-", color="C0", lw=1.0, ms=3, label="PSIM I_L6_")
        ax_i.plot((t_py   - t_py[0])  *1e6, iL_py,   "x--", color="C3", lw=1.0, ms=4, label="Python i_L")
        ax_i.set_ylabel(f"i_L [A]\n(seg {i+1}: t₀={t_psim[0]*1e3:.2f}ms)")
        ax_i.grid(alpha=0.3)
        if i == 0: ax_i.legend(loc="upper right", fontsize=8)

        ax_v.plot((t_psim - t_psim[0])*1e6, vC_psim, "o-", color="C0", lw=1.0, ms=3, label="PSIM udc_BB")
        ax_v.plot((t_py   - t_py[0])  *1e6, vC_py,   "x--", color="C3", lw=1.0, ms=4, label="Python v_C")
        ax_v.set_ylabel("u_dc [V]")
        ax_v.grid(alpha=0.3)
        if i == 0: ax_v.legend(loc="upper right", fontsize=8)

        d_iL = np.abs(iL_psim - iL_py).mean()
        d_vC = np.abs(vC_psim - vC_py).mean()
        diffs_iL.append(d_iL); diffs_vC.append(d_vC)
        print(f"Segment {i+1} (t₀={t_psim[0]*1e3:.3f}ms):  MAE i_L={d_iL:.4f} A   MAE u_dc={d_vC:.4f} V")

    axes[-1, 0].set_xlabel("Czas od początku segmentu [µs]")
    axes[-1, 1].set_xlabel("Czas od początku segmentu [µs]")
    fig.suptitle("Replay-test: Python startuje z warunków PSIM, każde okno = 20µs", y=1.001)
    plt.tight_layout()
    fname = "replay_test_bb_estim.png"
    plt.savefig(fname, dpi=110)
    print(f"\nMAE średnio: i_L = {np.mean(diffs_iL):.4f} A,  u_dc = {np.mean(diffs_vC):.4f} V")
    print(f"Zapisano wykres: {fname}")


if __name__ == "__main__":
    main()
