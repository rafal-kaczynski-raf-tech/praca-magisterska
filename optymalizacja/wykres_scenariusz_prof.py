"""Wykresy przebiegow dla scenariusza prof. Iwanskiego (4 warianty).

Dwa wykresy, kazdy 4 wiersze (ITAE / War.1 CurrentAware / War.2 CurrentOscillation
/ War.3 CurrentEffort) x 2 kolumny (napiecie u_dc, prad i_L):

  1) wykres_scenariusz_prof_pelny.png -- caly horyzont 0-300 ms, linie pionowe
     oznaczaja zdarzenia scenariusza (wylacz/wlacz urzadzenie, skok referencji).
  2) wykres_scenariusz_prof_zoom.png -- przyblizenie 140-210 ms (skok referencji
     240->160V @150ms + wylaczenie urzadzenia @200ms), pelna rozdzielczosc
     probek (bez decymacji) -- widac ksztalt przelaczania.

Uzywa nastaw PSO z pso_prof_*.npz (optymalizacja/pso_professor_scenario.py)
i sterownika z wlaczona korekta pradowa (current_correction=True, Eq.14-17).

Uruchomienie:  python -m optymalizacja.wykres_scenariusz_prof
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import default_config
from optymalizacja.scenarios import professor_scenario
from optymalizacja.analiza_scenariusz_prof import simulate, CASES

EVENTS = [
    (0.05, "OFF"), (0.10, "ON"), (0.15, "ref 240->160V"),
    (0.20, "OFF"), (0.25, "ON"),
]

COLORS = {
    "ITAE": "tab:gray",
    "CurrentAware": "tab:green",
    "CurrentOscillation": "tab:blue",
    "CurrentEffort": "tab:orange",
}


def _minmax_decimate(t: np.ndarray, y: np.ndarray, target_bins: int = 2000):
    """Decymacja min-max (bez artefaktow Moire) -- patrz wykres_przebiegi_wariantow.py."""
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


def _u_ref_arr(t: np.ndarray, u_ref0: float, ref_step_time: float, ref_step_value: float):
    arr = np.full_like(t, u_ref0)
    arr[t >= ref_step_time] = ref_step_value
    return arr


def _load_results():
    out = []
    for name, path in CASES.items():
        try:
            d = np.load(path)
        except FileNotFoundError:
            print(f"UWAGA: brak {path}, pomijam wariant {name} "
                  f"(uruchom najpierw pso_professor_scenario.py)")
            continue
        wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
        res = simulate(wi, fc)
        out.append((name, wi, fc, COLORS[name], res))
    return out


def _plot(results, u_ref0, ref_step_time, ref_step_value, win, out_path,
          title, decimate: bool) -> None:
    lo, hi = win
    fig, ax = plt.subplots(len(results), 2, figsize=(12, 10), sharex="col")

    for row, (name, wi, fc, color, res) in enumerate(results):
        t = res["t"]
        m = (t >= lo) & (t <= hi)
        tm = (t[m] - lo) * 1e3   # ms wzgledem lo

        u_ref = _u_ref_arr(t, u_ref0, ref_step_time, ref_step_value)
        ax[row, 0].plot(tm, u_ref[m], "k--", lw=1, alpha=0.5, label="u_ref")

        v = res["v_C"][m]
        iL = res["i_L"][m]
        if decimate:
            tm_v, v_dec = _minmax_decimate(tm, v)
            tm_i, i_dec = _minmax_decimate(tm, iL)
        else:
            tm_v, v_dec = tm, v
            tm_i, i_dec = tm, iL

        ax[row, 0].plot(tm_v, v_dec, color=color, lw=0.8)
        ax[row, 0].set_ylabel("u_dc [V]")
        ax[row, 0].grid(True, alpha=0.3)
        ax[row, 0].set_title(f"{name}  (wi={wi:.3f}, fc={fc:.0f} Hz)",
                              fontsize=9.5, loc="left")

        ax[row, 1].plot(tm_i, i_dec, color=color, lw=0.8)
        ax[row, 1].set_ylabel("i_L [A]")
        ax[row, 1].grid(True, alpha=0.3)

        for t_ev, label in EVENTS:
            if lo <= t_ev <= hi:
                tm_ev = (t_ev - lo) * 1e3
                ax[row, 0].axvline(tm_ev, color="red", lw=0.8, ls=":", alpha=0.6)
                ax[row, 1].axvline(tm_ev, color="red", lw=0.8, ls=":", alpha=0.6)
                if row == 0:
                    ax[row, 0].text(tm_ev, ax[row, 0].get_ylim()[1], label,
                                     fontsize=7, color="red", ha="center", va="bottom")

    ax[0, 0].legend(fontsize=8, loc="lower right")
    ax[-1, 0].set_xlabel(f"czas [ms] (0 = t={lo*1e3:.0f}ms)")
    ax[-1, 1].set_xlabel(f"czas [ms] (0 = t={lo*1e3:.0f}ms)")
    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    print(f"Zapisano: {out_path}")


def main() -> None:
    base = default_config()
    u_ref0 = base.controller.u_ref
    scn = professor_scenario()
    results = _load_results()
    if not results:
        print("Brak wynikow PSO -- nic do wykreslenia.")
        return

    _plot(results, u_ref0, scn.ref_step_time, scn.ref_step_value,
          win=(0.0, 0.30),
          out_path="optymalizacja/wykres_scenariusz_prof_pelny.png",
          title="Scenariusz prof. Iwanskiego -- caly horyzont (300 ms), "
                "korekta pradowa Eq.14-17 wlaczona (obwiednia min-max)",
          decimate=True)

    _plot(results, u_ref0, scn.ref_step_time, scn.ref_step_value,
          win=(0.140, 0.210),
          out_path="optymalizacja/wykres_scenariusz_prof_zoom.png",
          title="Zoom: skok referencji 240->160V (150ms) + wylaczenie "
                "urzadzenia (200ms), pelna rozdzielczosc probek",
          decimate=False)


if __name__ == "__main__":
    main()
