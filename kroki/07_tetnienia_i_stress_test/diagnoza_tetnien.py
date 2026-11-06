"""Diagnoza natury tetnien pradu: przelaczanie (fizyczne) czy cykl graniczny?

Zoom przebiegu i_L w stanie ustalonym + sygnal przelaczania s + FFT.
Porownanie z teoretycznym tetnieniem CCM:  di_L = V_in * D * T_sw / L.

Uruchomienie:  python -m optymalizacja.diagnoza_tetnien
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from src.config import default_config
from src.simulator import Simulator
from optymalizacja.scenarios import stress_scenario, STRESS_T_END


def simulate(wi, fc):
    base = default_config()
    cfg = replace(base, controller=replace(base.controller, wi=wi, fc_lpf=fc),
                  scenario=stress_scenario(), T_end=STRESS_T_END)
    return Simulator(cfg).run(), base


def main():
    # Optimum wariantu 1 (CurrentAware) na stress
    d = np.load("optymalizacja/pso_stress_currentaware.npz")
    wi, fc = float(d["wi_opt"]), float(d["fc_opt"])
    res, base = simulate(wi, fc)

    t = res["t"]
    iL = res["i_L"]
    v = res["v_C"]
    s = res["s"]

    L = base.converter.L
    V_in = base.converter.V_in

    # Okno stanu ustalonego w segmencie 280 V (0.040-0.060), po ustaleniu
    w_a, w_b = 0.0520, 0.0600
    win = (t >= w_a) & (t < w_b)
    tw, iw, sw, vw = t[win], iL[win], s[win], v[win]

    # Efektywna czestotliwosc przelaczania: liczba zboczy narastajacych / czas
    rising = np.sum((sw[1:] > 0.5) & (sw[:-1] <= 0.5))
    f_sw = rising / (w_b - w_a)

    # Duty (srednia s) i teoretyczne tetnienie CCM
    D = float(np.mean(sw))
    dt = t[1] - t[0]
    T_sw = 1.0 / f_sw if f_sw > 0 else np.nan
    diL_theory = V_in * D * T_sw / L     # przyblizenie tetnienia trojkatnego

    iL_pp = float(np.ptp(iw))
    iL_mean = float(np.mean(iw))

    # FFT pradu w oknie (usuwamy srednia)
    sig = iw - iL_mean
    n = len(sig)
    freqs = np.fft.rfftfreq(n, dt)
    amp = np.abs(np.fft.rfft(sig)) * 2.0 / n
    # dominujaca czestotliwosc (pomijamy DC)
    k = 1 + int(np.argmax(amp[1:]))
    f_dom = freqs[k]

    print("=" * 66)
    print("DIAGNOZA TETNIEN (wariant 1, stan ustalony 280 V)")
    print("=" * 66)
    print(f"  nastawy:        wi={wi:.3f}, fc={fc:.0f} Hz")
    print(f"  L = {L*1e3:.2f} mH,  V_in = {V_in:.0f} V")
    print(f"  prad sredni:    {iL_mean:.2f} A")
    print(f"  tetnienia p-p:  {iL_pp:.2f} A  ({100*iL_pp/iL_mean:.0f}% sredniego)")
    print(f"  f przelaczania: {f_sw/1e3:.1f} kHz  (duty D={D:.2f})")
    print(f"  f dominujaca w widmie iL: {f_dom/1e3:.1f} kHz")
    print(f"  tetnienie CCM teoria (V_in*D*T_sw/L): {diL_theory:.2f} A")
    print("=" * 66)
    verdict = ("PRZELACZANIE (fizyczne)" if abs(f_dom - f_sw) / f_sw < 0.3
               else "NISKA CZESTOTLIWOSC -> mozliwy cykl graniczny")
    print(f"  WERDYKT: dominanta widma ~ f_przelaczania ?  -> {verdict}")
    print("=" * 66)

    # --- Wykres ---
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))

    # 1) zoom 1 ms i_L + s
    zoom = (t >= 0.0540) & (t < 0.0550)
    tz = (t[zoom] - 0.0540) * 1e3   # ms
    ax[0].plot(tz, iL[zoom], color="tab:blue", lw=1.2, label="i_L [A]")
    ax0b = ax[0].twinx()
    ax0b.plot(tz, s[zoom], color="tab:red", lw=0.8, alpha=0.5,
              drawstyle="steps-post", label="s (przelacznik)")
    ax0b.set_ylim(-0.2, 1.2)
    ax0b.set_ylabel("s", color="tab:red")
    ax[0].set_xlabel("czas [ms] (okno 1 ms)")
    ax[0].set_ylabel("i_L [A]", color="tab:blue")
    ax[0].set_title(f"Zoom 1 ms: prad cewki vs przelaczanie  "
                    f"(f_sw={f_sw/1e3:.1f} kHz, tetn.={iL_pp:.1f} A p-p)")
    ax[0].grid(True, alpha=0.3)

    # 2) caly segment 280 V (stacjonarnosc tetnien)
    seg = (t >= 0.0400) & (t < 0.0600)
    ax[1].plot((t[seg]-0.0400)*1e3, iL[seg], color="tab:blue", lw=0.6)
    ax[1].axvspan((w_a-0.0400)*1e3, (w_b-0.0400)*1e3, color="tab:green", alpha=0.12,
                  label="okno analizy")
    ax[1].set_xlabel("czas [ms] od skoku na 280 V")
    ax[1].set_ylabel("i_L [A]")
    ax[1].set_title("Caly segment 280 V: tetnienia stacjonarne (nie zanikaja)")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc="upper right")

    # 3) FFT
    ax[2].plot(freqs/1e3, amp, color="tab:purple", lw=1.0)
    ax[2].axvline(f_sw/1e3, color="tab:red", ls="--", lw=1.0,
                  label=f"f_przelaczania {f_sw/1e3:.1f} kHz")
    ax[2].set_xlim(0, min(60, freqs.max()/1e3))
    ax[2].set_xlabel("czestotliwosc [kHz]")
    ax[2].set_ylabel("amplituda i_L [A]")
    ax[2].set_title("Widmo tetnien pradu (FFT, stan ustalony)")
    ax[2].grid(True, alpha=0.3)
    ax[2].legend(loc="upper right")

    fig.tight_layout()
    out = "diagnoza_tetnien.png"
    fig.savefig(out, dpi=130)
    print(f"Zapisano wykres: {out}")


if __name__ == "__main__":
    main()
