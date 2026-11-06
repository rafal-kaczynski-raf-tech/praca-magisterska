"""Przebiegi napiecia i pradu OSOBNO dla kazdej funkcji celu (uwaga prof. Iwanskiego).

Prof. Iwanski (elektryk/automatyk) poprosil o surowe przebiegi czasowe
napiecia i pradu dla kazdej z 4 nastaw (a nie zagregowane wykresy slupkowe),
bo "przebiegi pokazuja wszystko" -- w tym ewentualne artefakty przelaczania,
ktore widzial na wczesniejszym wykresie (diagnoza_tetnien.png, tylko wariant 1).

Generuje DWA wykresy, kazdy 4 wiersze (ITAE / Wariant 1 / Wariant 2 / Wariant 3)
x 2 kolumny (napiecie, prad):
  1) przebiegi_wariantow_transient.png -- okno przejsciowe wokol skoku
     referencji 240->280 V (t=40 ms), 6 ms, pokazuje CALA odpowiedz.
  2) przebiegi_wariantow_zoom.png -- waskie okno stanu ustalonego (1.2 ms),
     pokazuje detal przelaczania / ksztalt tetnien (tu widac artefakty).

Uruchomienie:  python -m optymalizacja.wykres_przebiegi_wariantow
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import default_config
from optymalizacja.analiza_ripple import simulate
from optymalizacja.scenarios import _STRESS_PULSES

# (etykieta, plik npz z optimum PSO, kolor)
CASES = [
    ("ITAE (odniesienie - tylko napiecie)", "optymalizacja/pso_stress_itae.npz", "tab:gray"),
    ("Wariant 1 (oba uchyby)",              "optymalizacja/pso_stress_currentaware.npz", "tab:green"),
    ("Wariant 2 (oscylacje 4 probek)",      "optymalizacja/pso_current_oscillation.npz", "tab:blue"),
    ("Wariant 3 (wielkosc pradu - effort)", "optymalizacja/pso_stress_currenteffort.npz", "tab:orange"),
]

# Okna czasowe (skok 240->280 V przy t=0.040 s wg scenarios._STRESS_PULSES)
TRANS_WIN = (0.037, 0.043)     # 6 ms: 3 ms przed skokiem + 3 ms po (cala odpowiedz)
ZOOM_WIN = (0.0560, 0.0572)    # 1.2 ms: stan ustalony (280 V), daleko od skokow


def _u_ref_arr(t: np.ndarray, u_ref0: float) -> np.ndarray:
    arr = np.full_like(t, u_ref0)
    for t_p, u_p in _STRESS_PULSES:
        arr[t >= t_p] = u_p
    return arr


def _minmax_decimate(t: np.ndarray, y: np.ndarray, target_bins: int = 700):
    """Decymacja min-max: dla kazdego 'binu' (docelowo ~1 px szerokosci) bierze
    min i max probki, zachowujac PRAWDZIWA obwiednie sygnalu bez artefaktow
    Moire, ktore powstaja przy naiwnym rysowaniu linii z >>1 probek/piksel.
    """
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
        # zachowaj kolejnosc czasowa min/max w binie (dla wiernego ksztaltu)
        if i_min <= i_max:
            t_out[2 * i], y_out[2 * i] = seg_t[i_min], seg_y[i_min]
            t_out[2 * i + 1], y_out[2 * i + 1] = seg_t[i_max], seg_y[i_max]
        else:
            t_out[2 * i], y_out[2 * i] = seg_t[i_max], seg_y[i_max]
            t_out[2 * i + 1], y_out[2 * i + 1] = seg_t[i_min], seg_y[i_min]
    return t_out, y_out


def _load_results():
    base = default_config()
    u_ref0 = base.controller.u_ref
    out = []
    for label, path, color in CASES:
        d = np.load(path)
        wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
        res = simulate(wi, fc)
        out.append((label, wi, fc, color, res))
    return out, u_ref0


def plot_transient(results, u_ref0, out_path):
    lo, hi = TRANS_WIN
    fig, ax = plt.subplots(len(results), 2, figsize=(12, 10), sharex="col")

    for row, (label, wi, fc, color, res) in enumerate(results):
        t = res["t"]
        m = (t >= lo) & (t <= hi)
        tm_full = (t[m] - lo) * 1e3

        u_ref = _u_ref_arr(t, u_ref0)
        ax[row, 0].plot(tm_full, u_ref[m], "k--", lw=1, alpha=0.5, label="u_ref")

        tm_v, v_dec = _minmax_decimate(tm_full, res["v_C"][m])
        ax[row, 0].plot(tm_v, v_dec, color=color, lw=0.8)
        ax[row, 0].set_ylabel("u_dc [V]")
        ax[row, 0].grid(True, alpha=0.3)
        ax[row, 0].set_title(f"{label}  (wi={wi:.3f}, fc={fc:.0f} Hz)", fontsize=9.5, loc="left")

        tm_i, i_dec = _minmax_decimate(tm_full, res["i_L"][m])
        ax[row, 1].plot(tm_i, i_dec, color=color, lw=0.8)
        ax[row, 1].set_ylabel("i_L [A]")
        ax[row, 1].grid(True, alpha=0.3)

    ax[0, 0].legend(fontsize=8, loc="lower right")
    ax[-1, 0].set_xlabel("czas [ms] (0 = t-3ms wzgledem skoku 240->280 V)")
    ax[-1, 1].set_xlabel("czas [ms] (0 = t-3ms wzgledem skoku 240->280 V)")
    fig.suptitle("Przebiegi napiecia i pradu -- skok referencji 240->280 V "
                 "(pelna odpowiedz, 6 ms; obwiednia min-max, bez artefaktow rysowania)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    print(f"Zapisano: {out_path}")


def plot_zoom(results, u_ref0, out_path):
    lo, hi = ZOOM_WIN
    fig, ax = plt.subplots(len(results), 2, figsize=(12, 10), sharex="col")

    for row, (label, wi, fc, color, res) in enumerate(results):
        t = res["t"]
        m = (t >= lo) & (t <= hi)
        tm = (t[m] - lo) * 1e3

        ax[row, 0].plot(tm, res["v_C"][m], color=color, lw=1.3, marker=".", ms=2)
        ax[row, 0].set_ylabel("u_dc [V]")
        ax[row, 0].grid(True, alpha=0.3)
        ax[row, 0].set_title(f"{label}  (wi={wi:.3f}, fc={fc:.0f} Hz)", fontsize=9.5, loc="left")

        iL = res["i_L"][m]
        ax[row, 1].plot(tm, iL, color=color, lw=1.3, marker=".", ms=2)
        ax[row, 1].set_ylabel("i_L [A]")
        ax[row, 1].grid(True, alpha=0.3)
        pp = float(np.ptp(iL))
        ax[row, 1].text(0.98, 0.06, f"p-p={pp:.2f} A", transform=ax[row, 1].transAxes,
                        ha="right", fontsize=8.5,
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

    ax[-1, 0].set_xlabel("czas [ms] (okno 1.2 ms, stan ustalony 280 V)")
    ax[-1, 1].set_xlabel("czas [ms] (okno 1.2 ms, stan ustalony 280 V)")
    fig.suptitle("Detal przelaczania w stanie ustalonym (280 V) -- ksztalt tetnien",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    print(f"Zapisano: {out_path}")


def main():
    results, u_ref0 = _load_results()
    plot_transient(results, u_ref0, "optymalizacja/przebiegi_wariantow_transient.png")
    plot_zoom(results, u_ref0, "optymalizacja/przebiegi_wariantow_zoom.png")


if __name__ == "__main__":
    main()
