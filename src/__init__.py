"""Warstwa OOP symulatora MF-BB.

Architektura (warstwy odpowiedzialnosci):
    config.SimulationConfig    - dataclass z parametrami (single source of truth)
    converter.Converter        - fizyka obwodu boost (krok Eulera)
    filters.ButterworthLPF     - biquad 2-go rzedu (bilinear)
    controller.MFBBController  - algorytm sterowania (estymator + LPF + decyzja)
    sampler.ZOHSampler         - 1-sample delay pomiarow
    simulator.Simulator        - orkiestracja petli + logowanie

Implementacja musi byc identyczna BIT-W-BIT z proceduralnym
demo_bangbang_estim.simulate() (sprawdzane przez tests/test_regression.py).
"""
