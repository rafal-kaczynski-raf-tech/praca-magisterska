# praca-magisterska — kod źródłowy

Implementacja **Bliźniaka Cyfrowego** (Digital Twin) dwukierunkowego przekształtnika boost DC-DC w języku **Python**, ze sterownikiem **Model-Free Bang-Bang (MF-BB)** strojonym **algorytmem PSO**. Kod stanowi część warsztatową pracy magisterskiej.

**Tekst pracy (LaTeX):** [github.com/rafal-kaczynski-raf-tech/praca-magisterska-latex](https://github.com/rafal-kaczynski-raf-tech/praca-magisterska-latex)

## Etapy projektu

| Krok | Opis | Status |
|---|---|---|
| **1.** Silnik fizyczny w otwartej pętli | Symulacja boost (V_in=100V, L=0.75mH, C=200µF, R=50Ω) metodą Eulera, dt=1µs, walidacja względem PSIM |ukończony |
| **2.** Sterownik MF-BB | Bezmodelowy bang-bang z filtrem LP na prądzie cewki, funkcja przełączająca S = K_v·(u_ref − u_C) + w_i·(i_des − i_L) | ukończony |
| **2.5.** Walidacja MF-BB vs PSIM | Replikacja C-blocku PSIM 1:1 (ZOH 1-sample delay, Euler dt=10µs, R_L=0.05Ω); stan ustalony t>6 ms: napięcie **0.07%**, tętnienia **1.6%**, prąd **0.00 A** różnicy względem PSIM | ukończony |
| **3.** PSO do strojenia parametrów | Automatyczna optymalizacja {K_v, w_i, ω_c} przez PSO z funkcją kosztu penalizującą overshoot + ripple | ⏳ w toku |
| **4.** Stress-testy | Małe pojemności filtra, AI vs strojenie ręczne | ⏳ planowane |
| **5.** Walidacja końcowa + dokumentacja | Pełne porównanie z PSIM, integracja z tekstem pracy | ⏳ planowane |

## Struktura plików

| Plik | Rola |
|---|---|
| `demo.py` | **Krok 1** — silnik fizyczny boost (otwarta pętla, walidacja vs `psim_wyniki.csv`) |
| `demo_bangbang.py` | **Krok 2** — sterownik MF-BB, skok napięcia 200 V → 240 V |
| `demo_bangbang_psim.py` | **Krok 2.5** — replikacja MF-BB 1:1 z C-blockiem PSIM (R_L=0); zapisuje `wykres_bb_psim_comparison.png` (roboczy) |
| `porownanie_bb_psim.py` | **Krok 2.5** — nakłada przebiegi Python na PSIM, zapisuje `porownanie_bb_python_vs_psim.png` (**wykres finalny dla promotora**) |
| `wykres_zoom_porownanie.py` | **Krok 2.5** — wykres dwupanelowy: full-sim (0–10 ms) + zoom 0–2 ms; zapisuje `wykres_zoom_porownanie.png` |
| `wykres_pwm_duty.py` | **Krok 2.5** — duty cycle PWM w oknie ruchomym 1 ms (uśrednia chaos histerezowy); zapisuje `wykres_pwm_duty.png` |
| `test_RL_sweep.py` | Diagnostyka — sweep rezystancji szeregowej cewki R_L |
| `test_dt_sweep.py` | Diagnostyka — sweep kroku całkowania dt |
| `test_first_samples.py` | Diagnostyka — analiza pierwszych 12 próbek stanu przejściowego |
| `psim_wyniki.csv` | Dane referencyjne PSIM dla **otwartej pętli** (Krok 1): kolumny Time, I(L7), I2, V2 |
| `psim_bb_wyniki.txt` | Dane referencyjne PSIM dla **MF-BB** (Krok 2.5): 9 kolumn (t, iL, ..., udc, ..., uref, pwm, iref) |
| `wyniki.txt` | Duplikat `psim_bb_wyniki.txt` (do uporządkowania) |
| `wykres_bb_psim_comparison.png` | Wykres roboczy (output `demo_bangbang_psim.py`) |
| `porownanie_bb_python_vs_psim.png` | **Wykres finalny dla promotora** (output `porownanie_bb_psim.py`) |
| `wykres_zoom_porownanie.png` | Wykres pomocniczy: full-sim + zoom 0–2 ms (output `wykres_zoom_porownanie.py`) |
| `wykres_pwm_duty.png` | Wykres pomocniczy: duty cycle PWM w oknie ruchomym 1 ms (output `wykres_pwm_duty.py`) |

## Wymagania

- **Python 3.10+**
- `numpy`
- `matplotlib`
- `pandas` (tylko `demo.py` — wczytywanie CSV)

Instalacja:
```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib pandas
```

## Uruchomienie

```bash
python demo.py                      # Krok 1: silnik fizyczny otwartej pętli
python demo_bangbang.py             # Krok 2: sterownik MF-BB
python demo_bangbang_psim.py        # Krok 2.5: replikacja PSIM → wykres_bb_psim_comparison.png
python porownanie_bb_psim.py        # Krok 2.5: wykres FINALNY dla promotora → porownanie_bb_python_vs_psim.png
python wykres_zoom_porownanie.py    # Krok 2.5: full-sim + zoom 0–2 ms      → wykres_zoom_porownanie.png
python wykres_pwm_duty.py           # Krok 2.5: duty cycle (okno 1 ms)      → wykres_pwm_duty.png
```

Każdy plik ma docstring na początku z dokładnym opisem działania.

## Branche

- `development` — bieżąca praca (najnowsze eksperymenty)
- `staging` — stabilne wyniki przed merge do `main`
- `main` — wersje punktowe odpowiadające etapom pracy

## Atrybucja źródeł algorytmów

Algorytm MF-BB oraz funkcja przełączająca pochodzą z artykułów grupy badawczej promotora pracy:

* **Artykuł bazowy (MF-BB):** F. R. Tatari, H. Bizhani, G. Iwański — *Short-horizon finite-state voltage control of bidirectional DC-DC converter with non-minimum phase dynamics*, Power Electronics and Drives, 2024.
* **Kontekst (Split-cost FS-MPC):** F. R. Tatari, H. Bizhani, G. Iwański — IEEE Journal of Emerging and Selected Topics in Power Electronics, 2025. DOI: [10.1109/JESTPE.2025.3535101](https://doi.org/10.1109/JESTPE.2025.3535101).

## Licencja

© Rafał Kaczyński, 2026. Kod źródłowy udostępniony na potrzeby pracy magisterskiej. Wszelkie prawa zastrzeżone do czasu określenia warunków otwartej licencji.
