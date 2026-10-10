"""Replay the full PSIM switching trace through the final converter model."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ConverterParams
from src.converter import Converter


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "psim_bb_estim_full.csv"
OUTPUT_PATH = ROOT / "replay_full_physics.png"
DT_PHYS = 0.5e-6


def main() -> None:
    data = pd.read_csv(CSV_PATH).sort_values("Time").reset_index(drop=True)
    time = data["Time"].to_numpy(dtype=float)
    pwm = data["pwmBB"].to_numpy(dtype=np.int8)
    psim_current = data["I(L6)"].to_numpy(dtype=float)
    psim_voltage = data["udc_BB"].to_numpy(dtype=float)

    intervals = np.diff(time)
    if not np.allclose(intervals, 10e-6, rtol=0.0, atol=1e-12):
        raise ValueError("Expected a continuous PSIM trace sampled every 10 us")

    converter = Converter(ConverterParams(i_L0=psim_current[0], v_C0=psim_voltage[0]))
    replay_current = np.empty_like(psim_current)
    replay_voltage = np.empty_like(psim_voltage)
    replay_current[0] = converter.i_L
    replay_voltage[0] = converter.v_C

    steps_per_sample = round(intervals[0] / DT_PHYS)
    for sample_index in range(1, len(time)):
        state = int(pwm[sample_index - 1])
        for _ in range(steps_per_sample):
            converter.step(state, DT_PHYS)
        replay_current[sample_index] = converter.i_L
        replay_voltage[sample_index] = converter.v_C

    current_error = replay_current - psim_current
    voltage_error = replay_voltage - psim_voltage
    print(f"Samples: {len(time)}; interval: {intervals[0] * 1e6:.3f} us; "
          f"replay duration: {(time[-1] - time[0]) * 1e3:.3f} ms")
    print(f"i_L: max abs error={np.max(np.abs(current_error)):.6g} A, "
          f"MAE={np.mean(np.abs(current_error)):.6g} A")
    print(f"u_dc: max abs error={np.max(np.abs(voltage_error)):.6g} V, "
          f"MAE={np.mean(np.abs(voltage_error)):.6g} V")

    time_ms = time * 1e3
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time_ms, psim_voltage, color="black", linewidth=1.2, label="PSIM")
    axes[0].plot(time_ms, replay_voltage, color="#1676b8", linewidth=0.9,
                 linestyle="--", label="Python: replay PWM")
    axes[0].set_ylabel("Napięcie wyjściowe $u_{dc}$ [V]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time_ms, psim_current, color="black", linewidth=1.2, label="PSIM")
    axes[1].plot(time_ms, replay_current, color="#1676b8", linewidth=0.9,
                 linestyle="--", label="Python: replay PWM")
    axes[1].set_xlabel("Czas [ms]")
    axes[1].set_ylabel("Prąd dławika $i_L$ [A]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    axes[1].set_xlim(time_ms[0], time_ms[-1])

    figure.suptitle("Test odtworzeniowy modelu fizycznego: PSIM i Python "
                    "(0,01–100 ms)")
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=180)
    print(f"Plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()