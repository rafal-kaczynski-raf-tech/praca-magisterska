import numpy as np # Importujemy bibliotekę NumPy (skrót 'np') do zaawansowanej matematyki, potrzebna nam do stworzenia tablic pamiętających tysiące ułamków sekund (nasza oś X).
import matplotlib.pyplot as plt # Importujemy Pyplot z Matplotlib (skrót 'plt'), to nasze "cyfrowe okienko Simview", dzięki niemu wyrysujemy na ekranie kolorowe krzywe.
import pandas as pd # Importujemy potężną bibliotekę Pandas (skrót 'pd'), jej jedynym zadaniem jest eleganckie wczytanie wyeksportowanego z PSIM pliku .csv.
import os # Importujemy wbudowany moduł 'os', który pozwala Pythonowi porozmawiać z systemem (użyjemy go żeby sprawdzić, czy plik CSV w ogóle leży w folderze).

# ==============================================================================
# 1. PARAMETRY FIZYCZNE I STEROWANIA (Cyfrowy klon Twojego układu z PSIM)
# ==============================================================================
V_in = 100.0 # Napięcie zasilacza V6 z lewej strony schematu. Ustaliliśmy, że offset to 100, a amplituda to 0, więc to po prostu bateria 100V prądu stałego (DC).
L = 0.75e-3 # Indukcyjność naszej głównej cewki 'l7'. Wzięte z oryginalnej nazwy pliku od profesora (0.75 mH). Jest to fizyczny magazyn energii pola magnetycznego.
R_L = 0.05 # Rezystancja drutu z jakiego zrobiona jest cewka. Odczytaliśmy te 0.05 Ohma z małego opornika szeregowego w PSIM. Zżera on nam ułamek mocy jako ciepło.
C = 1.5e-3 # Pojemność kondensatora na prawym końcu układu (1.5 mF). Przez mój wczesny błąd ustawiłem to na 200uF, teraz jest potężne i w końcu zgodne ze schematem.
R_load = 50.0 # Opór odbiornika. Po analizie skoku obciążenia (Tstep=0.2s) dowiedliśmy, że przez pierwsze 10 ms działa tylko ten jeden, lewy opornik na końcu układu.

f_sw = 2000.0 # Częstotliwość "klikania" tranzystorów (metronom) podana w Hertzach. Jest to wartość wpisana wewnątrz Twoich układów Gating Block po lewej stronie.
T_sw = 1.0 / f_sw # Okres jednego kliknięcia. Dzielimy 1 sekundę przez 2000 Hz i wychodzi nam, że pełen cykl przełączenia obu tranzystorów trwa dokładnie 0.0005 sekundy (0.5 ms).
D = 0.5 # Wypełnienie (Duty Cycle). Równe 0.5 (czyli 50%), ponieważ w Gating Blockach podaliśmy podział na równe połowy koła (punkty przełączania 0 do 180 stopni).

t_end = 0.01 # Moment, w którym symulacja ma się zatrzymać (10 milisekund). Odpowiada to dokładnie końcowej wartości czasu z Twoich screenów z modułu Simview w PSIM.
dt = 1e-6 # "Delta t", czyli mikrokrok czasowy (1 mikrosekunda). Komputer nie liczy czasu płynnie, lecz skacze klatka po klatce. Im mniejszy krok, tym idealniejsza symulacja.

# ==============================================================================
# 2. SYMULACJA NUMERYCZNA (Nasz własny silnik fizyczny krok-po-kroku)
# ==============================================================================
t_vals = np.arange(0, t_end, dt) # Tworzymy poziomą oś czasu (oś X). NumPy generuje nam miliony punktów od zera do 0.01s z krokiem co 1 mikrosekundę.
i_L_vals = np.zeros_like(t_vals) # Tworzymy pustą pamięć na prąd cewki (tyle samo punktów co czasu). Na razie są to same zera. Tu włożymy nasze wyliczenia (Sonda Current Probe).
v_C_vals = np.zeros_like(t_vals) # Tworzymy pustą pamięć na napięcie kondensatora. Tutaj po kolei będziemy wpisywać to, co odczytuje sonda V2 na schemacie w PSIM.

i_L = 0.0 # Stan początkowy prądu. Kiedy wciskasz "Run" w PSIM, cewka jest "pusta", więc nasz prąd na wejściu w czasie 0.0 to absolutne 0 Amperów.
v_C = 0.0 # Stan początkowy napięcia kondensatora. Tu też kondensator startuje rozładowany (0 Voltów). Tę wartość wcześniej błędnie miałem na 100.

for k in range(1, len(t_vals)): # Rozpoczynamy główną pętlę "czasową". Kod w środku tej pętli powtórzy się tysiące razy, dla każdej jednej mikrosekundy symulacji.
    t = t_vals[k-1] # Pobieramy z naszej gotowej listy czasu konkretną wartość ułamka sekundy dla obecnego obrotu pętli (np. t = 0.003501 sekundy).
    t_in_period = t % T_sw # Wyciągamy resztę z dzielenia czasu przez 0.5ms (modulo). Dzięki temu wiemy zawsze, na jakim etapie tyknięcia metronomu teraz jesteśmy.
    
    if t_in_period < D * T_sw: # Sprawdzamy czy obecny ułamek czasu tyknięcia jest mniejszy niż pierwsza połowa cyklu (czyli faza od 0 do 180 stopni z Gating Block).
        S = 1 # Jeśli tak: ustawiamy nasz wirtualny tranzystor na logiczne 1. Dolny tranzystor zamyka się do masy i "pompuje" energię z zasilacza w cewkę.
    else: # W przeciwnym wypadku (minęła pierwsza połowa i jesteśmy w fazie od 180 do 360 stopni z drugiego Gating Blocka)...
        S = 0 # Ustawiamy tranzystor na 0. Dolny tranzystor puszcza, otwiera się górny. Cewka przestaje ssać prąd z zasilacza i wypluwa zgromadzoną energię w kondensator.
        
    if S == 1: # Wchodzimy w prawa fizyki (Prawa Kirchhoffa). Jeśli układ ładuje cewkę:
        di_L = (V_in - i_L * R_L) / L # Wyliczamy z jaką prędkością rośnie prąd: Napięcie 100V minus opór cieplny na drutach (i*R) podzielone przez fizyczną wielkość cewki (L).
        dv_C = (-v_C / R_load) / C # Wyliczamy jak szybko napięcie spada: Kondensator oddaje zgromadzony prąd przez prawy opornik, więc jego wartość dzieli się przez pojemność.
    else: # Fizyka dla S=0 (Faza "Boost" - wystrzał w stronę kondensatora):
        di_L = (V_in - i_L * R_L - v_C) / L # Prędkość prądu teraz spada. Cewka napotyka potężny opór w postaci rosnącego napięcia ze wściekle ładującego się kondensatora (odjęto v_C).
        dv_C = (i_L - v_C / R_load) / C # Napięcie na kondensatorze gwałtownie rośnie. Kondensator połyka potężny prąd z uderzającej cewki (i_L), trochę ucieka przez odbiornik (R_load).
        
    i_L += di_L * dt # Metoda Eulera. Aktualizujemy nasz prąd rzeczywisty: nowy prąd cewki = stary prąd + (wyliczona prędkość jego zmian * nasza mikrosekunda czasu dt).
    v_C += dv_C * dt # To samo robimy dla napięcia. Nowe napięcie kondensatora = stare napięcie z ułamka sekundy wcześniej + (prędkość narastania napięcia * czas dt).
    
    i_L_vals[k] = i_L # Świeżutką, zaktualizowaną co do mikrosekundy wartość prądu cewki upychamy w naszej wielkiej tablicy pamięci pod odpowiednim punktem na osi czasu.
    v_C_vals[k] = v_C # Zaktualizowane napięcie kondensatora logujemy bezpiecznie na swojej liście, żeby Matplotlib miał z czego narysować zaraz wykres V2.

# ==============================================================================
# 3. WCZYTANIE PLIKU Z PSIM (Do weryfikacji w locie tzw. Hardware-in-the-Loop)
# ==============================================================================
plik_csv = 'psim_wyniki.csv' # Tworzymy zmienną z dokładną nazwą pliku, do którego zapisałeś te dzikie dane wyplute z okna Simview (musi być w tym samym folderze).
dane_psim_dostepne = False # Na start ustalamy flagę bezpieczeństwa na False, żeby kod nie próbował narysować czegoś z PSIM, jeśli zaraz okaże się, że zgubiłeś plik.

if os.path.exists(plik_csv): # Używamy funkcji systemu operacyjnego i sprawdzamy, czy plik o takiej nazwie dosłownie istnieje tam, gdzie jest ten skrypt Pythona.
    try: # Uruchamiamy tryb bezpieczny (try). Jeśli coś wybuchnie (np. zła nazwa sondy w pliku), program nie sypnie błędem w pół ekranu, tylko grzecznie ominie i napisze o co chodzi.
        psim_df = pd.read_csv(plik_csv) # Biblioteka Pandas tworzy obiekt DataFrame (tabelę w pamięci RAM) wciągając do niej wszystkie tysiące wierszy z pliku z PSIM.
        psim_df.columns = psim_df.columns.str.strip() # Magiczna sztuczka na błędy PSIM-a: ucina wszystkie przypadkowe spacje, którymi PSIM mógł otoczyć nazwy kolumn z sondami.
        
        t_psim = psim_df['Time'] # Wydobywamy z wczytanej tabeli Pandas dokładną oś czasu, którą stworzył PSIM (kolumna o nazwie 'Time') jako wzorzec referencyjny.
        v_psim = psim_df['V2'] # Z tabeli wyciągamy przebieg napięcia. W pliku musi to być kolumna 'V2' (tak jak ustaliłeś przed chwilą nową sondę).
        i_psim = psim_df['I(L7)'] # Z tej samej tabeli wyciągamy przebieg prądu z oryginalnej, zielonej cewki profesora (podpis w pliku to 'I(l7)').
        dane_psim_dostepne = True # Plik zczytano perfekcyjnie. Zapalamy zielone światło (zmieniamy flagę na True), program nałoży teraz te linie z PSIM na nasze wykresy.
        print("Pomyślnie wczytano dane z PSIM!") # Jeśli to widzisz w konsoli, oznacza to, że Pandas zeżarł plik i nazwy kolumn V2 i I(l7) zgadzają się co do litery.
    except Exception as e: # Jeśli blok 'try' nie wypali (plik ucięty, inne nazwy sond)... wchodzimy tu, żeby pokazać Ci grzeczny, tekstowy komunikat awarii w konsoli.
        print(f"Błąd podczas czytania CSV: {e}") # Printujemy w konsoli dokładnie techniczny błąd, przez który Pandas nie chciał wciągnąć Twoich kolumn z danymi.
        print("Upewnij się, że plik ma kolumny: Time, V2, I(l7)") # Printujemy podpowiedź inżynierską. Jeśli nie chce czytać, prawdopodobnie PSIM zrobił inną nazwę sondy na górze pliku txt.
else: # A jeśli funkcja systemowa sprawdzi folder i pliku w ogóle tam nie ma (np. nazwałeś go psim.csv zamiast psim_wyniki.csv)...
    print(f"UWAGA: Nie znaleziono pliku '{plik_csv}' w folderze ze skryptem.") # ...wypisuje w konsoli, że plik wyparował i rysuje dane z samej matematyki.
    print("Skrypt wygeneruje tylko wykresy z Pythona.") # Poinformowanie, że wykresy i tak się pokażą, tylko bez czarnych linii z PSIMa do porównania.

# ==============================================================================
# 4. RYSOWANIE WYKRESÓW (Nakładanie linii w Pythonie w okienku Matplotlib)
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True) # Tworzymy główne okno Windowsa/Maca o wymiarach 10x8 cali, podzielone na 2 panele (góra napięcie, dół prąd) ze wspólną osią osi (sharex).

# GÓRA: Panel z wykresem ładującego się NAPIĘCIA na kondensatorze.
ax1.plot(t_vals, v_C_vals, 'r', linewidth=3, label='Python (Model Matematyczny)') # Na panelu ax1 malujemy nasz wyliczony z pętli v_C_vals używając koloru czerwonego (r) i grubej linii 3 piksele.
if dane_psim_dostepne: # Sprawdzamy, czy nasze zielone światło z wgrywania z pliku się świeci (czyli plik PSIM działa poprawnie).
    ax1.plot(t_psim, v_psim, 'k--', linewidth=2, label='PSIM (Sonda V2)') # Jeśli tak, na tej samej osi nakładamy wykres z PSIM (v_psim). Używamy koloru czarnego (k) o przerywanej teksturze (--) i mniejszej grubości.
ax1.set_ylabel('Napięcie [V]') # Wstawiamy napis w pionie po lewej stronie tego okienka, żeby jasno było zaznaczone, że operujemy tu w Voltach.
ax1.set_title('Porównanie napięcia na kondensatorze (v_C)') # Piszemy u góry pierwszego wykresu wielki tytuł określający cel tego zestawienia.
ax1.grid(True) # Odpalamy szarą kratkę siatki w tle za wykresem. Zdecydowanie pomaga inżyniersko ocenić wartości bez mierzenia linijką ekranu.
ax1.legend() # Zlecamy narysowanie małego pudełka (legendy), które tłumaczy co to za czerwona gruba linia a co czarna przerywana, żeby było wiadomo co jest z Pythona a co z PSIM.

# DÓŁ: Panel z wykresem falującego, poszarpanego PRĄDU cewki.
ax2.plot(t_vals, i_L_vals, 'g', linewidth=3, label='Python (Model Matematyczny)') # Na dolnym panelu ax2 wylewamy nasz wyliczony prąd. Kolor zielony (g), gruby pędzel na 3 piksele. Pokaże nasze zęby prądowe.
if dane_psim_dostepne: # Znowu sprawdzamy, czy zassało poprawnie referencyjne pliki od profesora z CSV.
    ax2.plot(t_psim, i_psim, 'k--', linewidth=2, label='PSIM (Sonda I(l7))') # Rysujemy to, co wyciągnęło z prądowej kolumny I(l7) jako cieńszą (2px), czarną przerywaną kreskę. Ma wejść pod naszą zieloną.
ax2.set_ylabel('Prąd [A]') # Wstawiamy opis osi pionowej dolnego panelu - to jest prąd liczony w Amperach.
ax2.set_xlabel('Czas [s]') # Ustawiamy główną oś X widoczną na samym dole jako upływające sekundy. Ze względu na wczesny sharex, steruje ona górnym i dolnym obrazkiem naraz.
ax2.set_title('Porównanie prądu cewki (i_L)') # Wstawiamy duży, informacyjny tytuł również nad dolnym panelem symulacji prądowej.
ax2.grid(True) # Odpalamy znowu siatkę koordynatów w tle, żeby łatwo sprawdzić pik prądu w danym miejscu w czasie (bez tej siatki odczyty z pustego białego ekranu byłyby zgadywanką).
ax2.legend() # I znów budujemy małą tabelkę z opisem linii dla ułatwienia odczytu (co to za linia Pythonowa i PSIMowa w dolnej połówce okna).

plt.tight_layout() # Magiczny kod Matplotliba naprawiający wizualia. Sprawia on, że teksty, tytuły i osie nie najeżdżają i nie gryzą się na siebie nawzajem (automatyczne marginesy).
plt.show() # I ostateczny strzał. Ten guzik kompiluje wszystkie te powyższe wykresy do kupy i otwiera na Twoim ekranie dedykowane, wyskakujące okno z wizualizacją. Gotowe!