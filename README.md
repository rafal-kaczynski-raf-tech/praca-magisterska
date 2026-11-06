# praca-magisterska — kod źródłowy

Implementacja **Bliźniaka Cyfrowego** (Digital Twin) dwukierunkowego przekształtnika boost DC-DC w języku **Python**, ze sterownikiem **Model-Free Bang-Bang (MF-BB)** strojonym **algorytmem PSO**. Kod stanowi część warsztatową pracy magisterskiej.

**Tekst pracy (LaTeX):** [github.com/rafal-kaczynski-raf-tech/praca-magisterska-latex](https://github.com/rafal-kaczynski-raf-tech/praca-magisterska-latex)

## Kolejność badań

Historyczne wyniki, skrypty uruchamiające eksperymenty, dane wejściowe i odpowiadające im wykresy znajdują się w `kroki/`. Numer katalogu oznacza kolejność dochodzenia do końcowego rozwiązania.

| Krok | Katalog | Zakres |
|---|---|---|
| 1 | `kroki/01_silnik_fizyczny/` | Model boost w otwartej pętli i dane referencyjne PSIM |
| 2 | `kroki/02_sterownik_mf_bb/` | Pierwszy sterownik MF-BB |
| 3 | `kroki/03_walidacja_mf_bb_psim/` | Replikacja sterownika i walidacja MF-BB względem PSIM |
| 4 | `kroki/04_estymacja_pradu_i_walidacja/` | Estymacja prądu obciążenia i walidacja rozszerzonego sterownika |
| 5 | `kroki/05_walidacja_silnika_i_replay/` | Walidacja silnika fizycznego, testy replay i replika przebiegów z artykułu |
| 6 | `kroki/06_pso_i_funkcje_celu/` | Przeszukiwanie siatki, PSO i porównanie funkcji celu |
| 7 | `kroki/07_tetnienia_i_stress_test/` | Warianty funkcji celu, tętnienia prądu i scenariusze stresowe |
| 8 | `kroki/08_scenariusz_dynamiczny/` | Scenariusz dynamiczny i strojenie dla przebiegów promotora |
| 9 | `kroki/09_trend_mf_bb/` | Sterownik Trend MF-BB oraz funkcje tłumienia i progu |
| 10 | `kroki/10_porownanie_czestotliwosci/` | Porównanie częstotliwości próbkowania i metod sterowania |
| 11 | `kroki/11_optymalizacja_dwuetapowa/` | Dwuetapowa optymalizacja PSO w wariantach 1D i 2D |
| 12 | `kroki/12_badanie_wrazliwosci/` | Badanie wrażliwości optymalizacji dwuetapowej |
| 13 | `kroki/13_wplyw_pojemnosci/` | Wpływ pojemności kondensatora na regulator |
| 14 | `kroki/14_replay_koncowy/` | Końcowe porównanie replay modelu Python z fizyką PSIM |

Wspólny model i kod symulatora pozostają w `src/`, wspólne funkcje optymalizacji w `optymalizacja/`, a testy regresji w `tests/`. Katalogi kroków przechowują eksperymenty wraz z ich historycznymi wynikami.

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

Polecenia uruchamiaj z katalogu głównego repozytorium. Przykładowo:

```bash
PYTHONPATH=. python kroki/01_silnik_fizyczny/demo.py
PYTHONPATH=. python kroki/02_sterownik_mf_bb/demo_bangbang.py
PYTHONPATH=. python kroki/03_walidacja_mf_bb_psim/porownanie_bb_psim.py
.venv/bin/python tests/test_regression.py
```

Pozostałe eksperymenty i skrypty pomocnicze znajdują się w odpowiadających im katalogach `kroki/NN_nazwa/`.

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
