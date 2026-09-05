"""Replika scenariusza Fig. 8 z artykulu IEEE (Tatari/Bizhani/Iwanski, JESTIE),
sekcja VI-A-1 "Proposed BB Control Performance" -- najblizszy "kanoniczny"
test opisany wprost w tekscie (parametry + zdarzenia jednoznacznie podane).

Zdarzenia (wg tekstu artykulu):
  t=0.00s  start, u_ref=250V, R=41.6 Ohm (CCM)
  t=0.15s  skok referencji 250V -> 150V
  t=0.30s  skok obciazenia R: 41.6 -> 208 Ohm (przejscie CCM -> DCM)

Parametry fizyczne (Table I): vs=100V, L=750uH, C=1500uF, Ts=20us,
imax_L=20A. Waga lambda wg empirycznej reguly artykulu:
lambda(C) = 7.48*C[uF]^-0.547 (C=1500uF -> lambda~=0.137). fc_lpf=200Hz
("effective compromise", wprost podane w tekscie). current_correction=True
(pelny model Eq.14-17, bo to jest wlasnie "proposed MBB" z artykulu).

WAZNE ZASTRZEZENIE: to NIE jest bit-w-bit odtworzenie wykresu z artykulu --
nie mamy dostepu do surowego kodu/danych autorow, wiec nie da sie
zweryfikowac identycznosci probka-po-probce. To, co robimy, to symulacja
NASZYM silnikiem (ten sam model matematyczny obwodu boost + ten sam
algorytm sterowania Eq.14-17) z TYMI SAMYMI parametrami liczbowymi
podanymi w Table I artykulu, zeby sprawdzic, czy JAKOSCIOWO (ksztalt,
ustalanie sie napiecia, redukcja tetnien, brak uchybu ustalonego)
zachowanie zgadza sie z opisem tekstowym w artykule.

Uruchomienie:  python -m optymalizacja.replika_artykul_fig8
"""
from __future__ import annotations
import numpy as np
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import SimulationConfig, ConverterParams, ControllerParams
from src.converter import Converter
from src.simulator import Simulator
from optymalizacja.scenarios import (
    article_fig8_scenario, ARTICLE_T_END, ARTICLE_V_IN, ARTICLE_L,
    ARTICLE_C, ARTICLE_R_INITIAL, ARTICLE_R_AFTER, ARTICLE_TS,
    ARTICLE_I_MAX, ARTICLE_FC_LPF, ARTICLE_U_REF0, ARTICLE_U_REF1,
    ARTICLE_REF_STEP_TIME, ARTICLE_LOAD_STEP_TIME, ARTICLE_LAMBDA,
)

SEG_EDGES = [0.0, ARTICLE_REF_STEP_TIME, ARTICLE_LOAD_STEP_TIME, ARTICLE_T_END]
SEG_R = [ARTICLE_R_INITIAL, ARTICLE_R_INITIAL, ARTICLE_R_AFTER]
SEG_UREF = [ARTICLE_U_REF0, ARTICLE_U_REF1, ARTICLE_U_REF1]
EVENTS = [(ARTICLE_REF_STEP_TIME, "ref 250->150V"),
          (ARTICLE_LOAD_STEP_TIME, "R 41.6->208 Ohm")]


def build_config() -> SimulationConfig:
    conv = ConverterParams(
        V_in=ARTICLE_V_IN, L=ARTICLE_L, C=ARTICLE_C,
        R_load=ARTICLE_R_INITIAL, v_C0=ARTICLE_V_IN, i_L0=0.0,
    )
    ctrl = ControllerParams(
        u_ref=ARTICLE_U_REF0, T_s_ctrl=ARTICLE_TS, wi=ARTICLE_LAMBDA,
        i_max=ARTICLE_I_MAX, fc_lpf=ARTICLE_FC_LPF, current_correction=True,
    )
    return SimulationConfig(
        converter=conv, controller=ctrl, dt_phys=0.5e-6,
        T_end=ARTICLE_T_END, scenario=article_fig8_scenario(),
    )


class DiodeConverter(Converter):
    """Wariant TYLKO do tego eksperymentu: blokada diody (i_L nie moze byc
    ujemny), jak w prawdziwym ukladzie diodowym z artykulu (S, C, L i D -
    dioda D wprost w schemacie, sekcja II). Nasz domyslny `Converter` jest
    SYNCHRONICZNY (R_ds_on=0, brak osobnych diod - potwierdzone w PSIM,
    patrz psim_validation.md) i CELOWO dopuszcza ujemny i_L - to jest
    zwalidowany model calej pracy, NIE zmieniamy go globalnie. Ta klasa
    zyje wylacznie w tym skrypcie, do jednorazowego porownania z Fig.8.

    Gdy przewidywany i_L(k+1) wyszedlby ponizej zera, dioda przerywa
    przewodzenie w tym kroku: i_L=0, a kondensator zasila sam odbiornik
    (i_C = -v_C/R_load), zamiast standardowej formuly synchronicznej.
    """

    def step(self, s: int, dt: float) -> None:
        p = self.p
        v_L = p.V_in - (1.0 - s) * self.v_C - p.R_L * self.i_L
        i_C_sync = (1.0 - s) * self.i_L - self.v_C / self.R_load
        i_L_new = self.i_L + v_L / p.L * dt
        if i_L_new < 0.0:
            i_L_new = 0.0
            i_C = -self.v_C / self.R_load
        else:
            i_C = i_C_sync
        self.i_L = i_L_new
        self.v_C = self.v_C + i_C / p.C * dt


def simulate() -> dict:
    return Simulator(build_config()).run()


def simulate_diode() -> dict:
    """Ta sama konfiguracja co simulate(), ale z blokada diody (DCM jak
    w prawdziwym ukladzie diodowym z artykulu)."""
    sim = Simulator(build_config())
    sim.converter = DiodeConverter(build_config().converter)
    return sim.run()


def analyse(res: dict) -> list[dict]:
    """Uchyb napiecia + tetnienia pradu w stanie ustalonym (ostatnie 40% seg.),
    plus porownanie estymowanego a rzeczywistego pradu obciazenia (jak w
    opisie Fig.8: "the estimated output current closely follows the
    measured value")."""
    t, v, iL = res["t"], res["v_C"], res["i_L"]
    t_c, iout_filt = res["t_ctrl"], res["iout_filt"]

    rows = []
    for a, b, R, u_ref in zip(SEG_EDGES[:-1], SEG_EDGES[1:], SEG_R, SEG_UREF):
        dur = b - a
        w0 = a + 0.6 * dur
        m = (t >= w0) & (t < b)
        mc = (t_c >= w0) & (t_c < b)
        if m.sum() < 5:
            continue
        v_w, iL_w = v[m], iL[m]
        v_err = float(np.mean(v_w) - u_ref)
        iL_mean = float(np.mean(iL_w))
        iL_pp = float(np.ptp(iL_w))
        iL_ripple_pct = 100.0 * iL_pp / abs(iL_mean) if abs(iL_mean) >= 0.05 else float("nan")

        i_load_actual = float(np.mean(v_w) / R)          # iLoad = vout/R
        i_load_est = float(np.mean(iout_filt[mc])) if mc.sum() > 0 else float("nan")

        rows.append(dict(
            segment=f"[{a:.2f},{b:.2f})s", u_ref=u_ref, R=R,
            v_err_V=v_err, iL_mean_A=iL_mean, iL_ripple_pp_A=iL_pp,
            iL_ripple_pct=iL_ripple_pct,
            i_load_actual_A=i_load_actual, i_load_est_A=i_load_est,
        ))
    return rows


def print_report(rows: list[dict]) -> None:
    print(f"\nlambda (wi) = {ARTICLE_LAMBDA:.4f}  (wg lambda(C)=7.48*C[uF]^-0.547, C=1500uF)")
    print(f"{'segment':<14}{'u_ref':>7}{'R':>8}{'v_err[V]':>10}"
          f"{'iL_mean[A]':>12}{'ripple_pp[A]':>14}{'ripple[%]':>11}"
          f"{'iLoad_akt[A]':>14}{'iLoad_est[A]':>14}")
    for r in rows:
        print(f"{r['segment']:<14}{r['u_ref']:>7.0f}{r['R']:>8.1f}"
              f"{r['v_err_V']:>10.3f}{r['iL_mean_A']:>12.3f}"
              f"{r['iL_ripple_pp_A']:>14.3f}{r['iL_ripple_pct']:>11.2f}"
              f"{r['i_load_actual_A']:>14.3f}{r['i_load_est_A']:>14.3f}")


def _minmax_decimate(t: np.ndarray, y: np.ndarray, target_bins: int = 2000):
    n = len(t)
    if n <= target_bins * 2:
        return t, y
    bin_size = n // target_bins
    n_bins = n // bin_size
    t_out = np.empty(n_bins * 2)
    y_out = np.empty(n_bins * 2)
    for i in range(n_bins):
        seg_t = t[i * bin_size:(i + 1) * bin_size]
        seg_y = y[i * bin_size:(i + 1) * bin_size]
        i_min = np.argmin(seg_y)
        i_max = np.argmax(seg_y)
        if i_min <= i_max:
            t_out[2 * i], y_out[2 * i] = seg_t[i_min], seg_y[i_min]
            t_out[2 * i + 1], y_out[2 * i + 1] = seg_t[i_max], seg_y[i_max]
        else:
            t_out[2 * i], y_out[2 * i] = seg_t[i_max], seg_y[i_max]
            t_out[2 * i + 1], y_out[2 * i + 1] = seg_t[i_min], seg_y[i_min]
    return t_out, y_out


def _u_ref_arr(t: np.ndarray) -> np.ndarray:
    arr = np.full_like(t, ARTICLE_U_REF0)
    arr[t >= ARTICLE_REF_STEP_TIME] = ARTICLE_U_REF1
    return arr


def plot(res: dict, win, out_path: str, title: str, decimate: bool) -> None:
    lo, hi = win
    t = res["t"]
    m = (t >= lo) & (t <= hi)
    tm = (t[m] - lo) * 1e3

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    u_ref = _u_ref_arr(t)
    ax[0].plot(tm, u_ref[m], "k--", lw=1, alpha=0.5, label="u_ref")
    v = res["v_C"][m]
    iL = res["i_L"][m]
    if decimate:
        tm_v, v_dec = _minmax_decimate(tm, v)
        tm_i, i_dec = _minmax_decimate(tm, iL)
    else:
        tm_v, v_dec = tm, v
        tm_i, i_dec = tm, iL
    ax[0].plot(tm_v, v_dec, color="tab:blue", lw=0.9)
    ax[0].set_ylabel("u_dc [V]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8, loc="lower right")

    ax[1].plot(tm_i, i_dec, color="tab:orange", lw=0.8)
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_xlabel(f"czas [ms] (0 = t={lo*1e3:.0f}ms)")
    ax[1].grid(True, alpha=0.3)

    for t_ev, label in EVENTS:
        if lo <= t_ev <= hi:
            tm_ev = (t_ev - lo) * 1e3
            ax[0].axvline(tm_ev, color="red", lw=0.8, ls=":", alpha=0.6)
            ax[1].axvline(tm_ev, color="red", lw=0.8, ls=":", alpha=0.6)
            ax[0].text(tm_ev, ax[0].get_ylim()[1], label, fontsize=7.5,
                       color="red", ha="center", va="bottom")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    print(f"Zapisano: {out_path}")


def plot_compare_iL(res_sync: dict, res_diode: dict, win, out_path: str) -> None:
    """Nakladka i_L: model synchroniczny (nasz domyslny, zwalidowany vs PSIM)
    vs model z blokada diody (jak w ukladzie z artykulu) -- pelna rozdzielczosc,
    zeby bylo widac zniknięcie ujemnych wychyleń prądu."""
    lo, hi = win
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for res, label, color in [(res_sync, "synchroniczny (domyslny, wg PSIM)", "tab:orange"),
                               (res_diode, "z blokada diody (jak w artykule)", "tab:green")]:
        t = res["t"]
        m = (t >= lo) & (t <= hi)
        tm = (t[m] - lo) * 1e3
        ax.plot(tm, res["i_L"][m], color=color, lw=0.9, label=label)
    ax.axhline(0.0, color="k", lw=0.7, alpha=0.5)
    ax.set_ylabel("i_L [A]")
    ax.set_xlabel(f"czas [ms] (0 = t={lo*1e3:.0f}ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.suptitle("Porownanie i_L: model synchroniczny vs model z blokada diody",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=130)
    print(f"Zapisano: {out_path}")


def main() -> None:
    res = simulate()
    rows = analyse(res)
    print("=== Model synchroniczny (domyslny silnik pracy, zwalidowany vs PSIM) ===")
    print_report(rows)

    plot(res, win=(0.0, ARTICLE_T_END),
         out_path="optymalizacja/wykres_replika_artykul_fig8_pelny.png",
         title="Replika Fig.8 artykulu (Table I + Eq.14-17, lambda=lambda(C)) "
               "-- caly horyzont, obwiednia min-max",
         decimate=True)

    plot(res, win=(0.28, 0.36),
         out_path="optymalizacja/wykres_replika_artykul_fig8_zoom.png",
         title="Zoom: skok obciazenia 41.6->208 Ohm @300ms (przejscie CCM->DCM), "
               "pelna rozdzielczosc",
         decimate=False)

    res_d = simulate_diode()
    rows_d = analyse(res_d)
    print("\n=== Wariant eksperymentalny: blokada diody (i_L>=0), jak w artykule ===")
    print_report(rows_d)

    plot(res_d, win=(0.0, ARTICLE_T_END),
         out_path="optymalizacja/wykres_replika_artykul_fig8_diode_pelny.png",
         title="Wariant z blokada diody (i_L>=0) -- caly horyzont, obwiednia min-max",
         decimate=True)

    plot(res_d, win=(0.28, 0.36),
         out_path="optymalizacja/wykres_replika_artykul_fig8_diode_zoom.png",
         title="Wariant z blokada diody -- zoom skoku obciazenia @300ms",
         decimate=False)

    plot_compare_iL(res, res_d, win=(0.28, 0.36),
                    out_path="optymalizacja/wykres_replika_artykul_fig8_compare_iL.png")
    plot_compare_iL(res, res_d, win=(0.145, 0.175),
                    out_path="optymalizacja/wykres_replika_artykul_fig8_compare_iL_refstep.png")


if __name__ == "__main__":
    main()
