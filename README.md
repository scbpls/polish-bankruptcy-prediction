# Predykcja upadłości polskich przedsiębiorstw

## 1. Cel biznesowy i definicja problemu

**Cel:** Stworzenie modelu uczenia maszynowego przewidującego bankructwo firmy na podstawie wskaźników finansowych. Analiza dotyczy historycznych danych polskich przedsiębiorstw.

**Problem biznesowy:**
Standardowe podejście do klasyfikacji (maksymalizacja trafności) jest nieskuteczne w przypadku predykcji bankructwa, ponieważ:

1.  **Niezbalansowane dane:** Bankructwa stanowią margines wszystkich firm (~5%). Model, który zawsze przewiduje "firma zdrowa", miałby skuteczność 95%, ale byłby bezużyteczny.

2.  **Asymetria kosztów:**
    - **Błąd I rodzaju (False Negative):** Model uznaje bankruta za firmę zdrową. **Koszt:** Utrata kapitału (np. niespłacony kredyt). Jest to błąd krytyczny.
    - **Błąd II rodzaju (False Positive):** Model uznaje zdrową firmę za zagrożoną. **Koszt:** Czas pracy analityka na weryfikację. Jest to koszt akceptowalny.

**Metryka sukcesu:**

- Priorytet: **Czułość (recall)** - celujemy w wykrycie min. 85% rzeczywistych bankructw.
- Kontrola: **Pole pod krzywą ROC (ROC AUC)** - jako miara ogólnej zdolności modelu do mierzenia ryzyka.

### 1.1. Wybór zakresu danych

Zbiór danych **Polish Companies Bankruptcy** zawiera 5 podzbiorów (od 1 do 5 lat przed upadkiem). Do analizy wybrano **rok 3**, ponieważ:

1.  **Kompromis biznesowy:**
    - _Rok 5 (1 rok przed upadkiem):_ Modele są najdokładniejsze, ale sygnał przychodzi zbyt późno na reakcję. Firma jest już zazwyczaj niewypłacalna.
    - _Rok 1 (5 lat przed upadkiem):_ Horyzont czasowy jest zbyt odległy, dane zawierają zbyt dużo szumu, co generuje wiele fałszywych alarmów.
    - _Rok 3 (złoty środek):_ 3 lata to optymalny czas, aby bank mógł podjąć działania prewencyjne (restrukturyzacja, wstrzymanie finansowania), a dane finansowe wykazują już wyraźne symptomy pogorszenia płynności.

2.  **Poprawność metodologiczna:** Nie połączono wszystkich lat w jeden zbiór, aby uniknąć naruszenia niezależności obserwacji (ta sama firma występująca wielokrotnie w zbiorze treningowym mogłaby doprowadzić do przeuczenia modelu na specyfice konkretnych podmiotów zamiast na ogólnych wzorcach).

## 2. Eksploracja i przygotowanie danych

### 2.1. Eksploracyjna analiza danych (EDA)

Przed jakąkolwiek ingerencją w dane, przeprowadzono ich analizę statystyczną:

1.  **Analiza liczebności klas:** Zweryfikowano balans zmiennej celu. Potwierdzono silne niezbalansowanie - bankruci stanowią jedynie margines (~4,7%) całego zbioru, co zdeterminowało późniejsze użycie warstwowania (`stratify`) przy podziale oraz ważenia klas w modelach.

2.  **Analiza korelacji:** Zbadano macierz korelacji dla 10 najistotniejszych cech, wizualizując wyniki za pomocą wykresu motylkowego.
    - **Wynik:** Wykres ujawnił wyraźną polaryzację. Wskaźniki zadłużenia korelują z bankructwem **dodatnio** (im wyższy dług, tym większe ryzyko), natomiast wskaźniki rentowności korelują **ujemnie** (im wyższy zysk, tym bezpieczniejsza firma).
    - **Wniosek:** Mimo widocznej współliniowości cech, nie usunięto ich na tym etapie, ponieważ planowane modele drzewiaste (XGBoost) radzą sobie z tym zjawiskiem automatycznie.

### 2.2. Podział i potok przetwarzania

Po analizie wstępnej zbudowano potok przetwarzania (ETL) zaprojektowany tak, aby uniknąć wycieku danych:

1.  **Podział danych:**
    - Zbiór danych podzielono na treningowy i testowy w proporcji **80/20**.
    - Użyto parametru `stratify=y`, aby zachować identyczny odsetek bankrutów w obu podzbiorach.

2.  **Czyszczenie i wybór cech:**
    - Usunięto cechę `Attr37` _(aktywa obrotowe - zapasy) / zobowiązania długoterminowe_ ze względu na krytyczną liczbę brakujących danych (~45%). Liczba cech spadła z 64 do **63**.

3.  **Imputacja i skalowanie danych:**
    - Pozostałe braki (`NaN`) uzupełniono **medianą**, która jset odporna na wartości odstające.
    - Zastosowano `StandardScaler` do standaryzacji zmiennych.
    - Parametry imputera i skalera wyliczono wyłącznie na zbiorze treningowym, a następnie zaaplikowano do zbioru testowego.

### 2.3. Charakterystyka zbioru po przetworzeniu

Analizowany zbiór danych (rok 3) składa się łącznie z **10 503** przedsiębiorstw. Każda firma początkowo była opisana 64 wskaźnikami finansowymi.

W wyniku powyższych operacji uzyskano następującą strukturę danych gotowych do modelowania:

- **Zbiór treningowy:** 8402 firmy.
- **Zbiór testowy:** 2101 firm.
- **Liczba cech:** 63 wskaźniki.
- **Balans klas:** W wydzielonym zbiorze testowym na 2101 firm znajduje się tylko **99** bankrutów (~4,7%), co stanowi istotne wyzwanie dla modeli klasyfikacyjnych.

### 2.4. Weryfikacja poprzez PCA

Na przygotowanych danych przeprowadzono analizę PCA (Principal Component Analysis) w celach wizualizacyjnych.

- **Wynik:** Wizualizacja w przestrzeni 2D wykazała silne nakładanie się klas, co potwierdziło, że problem nie jest liniowy i wymaga wykorzystania nieliniowych modeli.
- **Decyzja:** Zrezygnowano z użycia PCA w finalnym treningu modelu, aby zachować pełną **interpretowalność cech** (biznes musi wiedzieć, który konkretnie wskaźnik finansowy decyduje o ryzyku, a nie abstrakcyjna składowa).

## 3. Eksperymenty modelowe

Przeprowadzono testy porównawcze trzech klas algorytmów. Wyniki śledzono przy użyciu platformy **Weights & Biases (WandB)**.

1.  **Baseline: Regresja logistyczna (logistic regression):**
    - **Konfiguracja:** Zastosowano `class_weight="balanced"`.
    - **Wynik:** ROC AUC ~0,74, Recall ~0,72, F1 Score ~0,17.
    - **Wniosek:** Zastosowanie parametru równoważenia klas wymusiło na modelu agresywne wykrywanie bankrutów, co dało wysoki poziom czułości (~72%). Jednak bardzo niska miara F1 zdradza, że model generuje ogromną liczbę fałszywych alarmów (błędów I rodzaju). Oznacza to, że choć wykrywa bankrutów, to przy okazji błędnie oznacza jako zagrożone wiele zdrowych firm, co czyni go mało praktycznym w użyciu.

2.  **Las losowy (random forest):**
    - **Konfiguracja:** Zastosowano `class_weight="balanced"` oraz optymalizację hiperparametrów GridSearch (liczbę drzew `n_estimators`, głębokość drzew `max_depth` i minimalną liczbę próbek w liściu `min_samples_leaf`).
    - **Wynik:** ROC AUC ~0,88, Recall ~0,12, F1 Score ~0,21.
    - **Wniosek:** Model poprawił jakość mierzenia ryzyka (wyższe ROC AUC niż dla modelu Baseline), ale okazał się niezwykle konserwatywny. Mimo ważenia klas, przy standardowym progu decyzyjnym (0,5), las losowy ignoruje większość zagrożeń (wykrywa tylko ~12% bankrutów). Taki wynik jest biznesowo nieakceptowalny (zbyt duże ryzyko przeoczenia upadłości).

3.  **Extreme Gradient Boosting (XGBoost):**
    - **Konfiguracja:** Zastosowano `scale_pos_weight` (waga dla klasy mniejszościowej) oraz optymalizację hiperparametrów GridSearch (liczbę drzew `n_estimators`, głębokość drzew `max_depth` i szybkość uczenia `learning_rate`).
    - **Wynik:** ROC AUC ~0,92, Recall ~0,47, F1 Score ~0,58.
    - **Wniosek:** Model osiągnął najwyższy wynik ROC AUC, co oznacza, że najlepiej rozróżnia firmy zdrowe od bankrutów. Choć czułość przy domyślnym progu (~47%) jest niższy niż w dla modelu Baseline, to wysokie ROC AUC sugeruje duży modelu. Oznacza to, że _wiedza_ o bankructwach jest w modelu, ale wymaga _wydobycia_ poprzez kalibrację progu decyzyjnego (co uczyniono w kolejnym kroku).

## 4. Optymalizacja progu

Domyślny model XGBoost (próg decyzyjny = 0,50) osiągał wysokie ROC AUC, ale niską czułość (~47%). Oznaczało to, że system przepuszczał ponad połowę bankrutów, klasyfikując ich jako bezpiecznych.

**Działanie:**

- Przeprowadzono analizę krzywej precyzja-czułość.
- Przesunięto próg decyzyjny z 0,50 na **optymalny poziom wyliczony na zbiorze testowym (~0,018)**.

**Efekt biznesowy:**

- **Zysk:** Czułość wzrosła z \~47% do **\~85%**.
- **Koszt:** Spadek precyzji (więcej fałszywych alarmów).
- **Uzasadnienie:** Akceptujemy wyższy koszt weryfikacji manualnej (koszt operacyjny), aby zminimalizować ryzyko udzielenia _złego_ kredytu (koszt kapitałowy).

## 5. Wyniki końcowe

| Model                              | ROC AUC  |  Recall  | F1 Score | Status                                                      |
| :--------------------------------- | :------: | :------: | :------: | :---------------------------------------------------------- |
| **Baseline (logistic regression)** |   0,74   |   0,72   |   0,17   | **Odrzucony** (przeciętna jakość i dużo fałszywych alarmów) |
| **Random forest (Tuned)**          | **0,88** |   0,12   |   0,21   | **Odrzucony** (wysoki potencjał, ale zbyt konserwatywny)    |
| **XGBoost (Tuned)**                | **0,92** |   0,47   | **0,58** | Dobry, ale słabe bezpieczeństwo                             |
| **XGBoost (Tuned & Optimized)**    | **0,92** | **0,85** |   0,33   | **Zatwierdzony**                                            |

_Uwaga: ROC AUC dla obu wersji XGBoost jest identyczne, ponieważ model matematycznie jest ten sam - zmienił się tylko próg decyzyjny (threshold)._

## 6. Istotność cech

Analiza istoności cech wskazała kluczowe wskaźniki finansowe:

1.  **`Attr26` _(zysk netto + amortyzacja) / zobowiązania ogółem_:** Zdolność firmy do spłaty całego długu z bieżącej gotówki. Kluczowy miernik wypłacalności.
2.  **`Attr34` _koszty operacyjne / zobowiązania ogółem_:** Wskaźnik obciążenia firmy kosztami. U bankrutów jest on nienaturalnie wysoki w relacji do zadłużenia.
3.  **`Attr24` _zysk brutto (w 3 lata) / aktywa ogółem_:** Długoterminowa rentowność majątku. Wskazuje, czy firma trwale traci na wartości.
4.  **`Attr27` _zysk z działalności operacyjnej / koszty finansowe_:** Zdolność do obsługi odsetek z podstawowej działalności (pokrycie odsetek).
5.  **`Attr39` _zysk na sprzedaży / przychody ze sprzedaży_:** Rentowność sprzedaży (ROS). Niska marża czyni firmę bezbronną wobec kosztów długu.

**Wniosek:** Bankructwo w badanym zbiorze nie wynika z nagłego braku gotówki, lecz z **trwałej niezdolności do obsługi długu** (`Attr26`, `Attr27`). Firmy te mają **zbyt niskie marże** (`Attr39`) przy **zbyt wysokich kosztach operacyjnych** (`Attr34`), by udźwignąć swoje zobowiązania. Obecność wskaźnika długoterminowej rentowności (`Attr24`) potwierdza, że jest to **stan chroniczny** - firmy te wykazują trwałą, wieloletnią nieefektywność w generowaniu zysku ze swojego majątku.

## 7. Podsumowanie techniczne

- **Technologie:** Python 3.12, Scikit-Learn, XGBoost, WandB, Pandas, NumPy, Joblib, SciPy, Matplotlib, Seaborn, Python-dotenv.
- **Artefakty projektu:**
  - Wszystkie modele zapisano w formacie `.joblib`.
  - Plik `Threshold_Config_Year3.json` zawiera wyliczony optymalny próg - jest to niezbędny element wdrażania modelu na produkcji.
  - Pełna historia eksperymentów dostępna w dashboardzie Weights & Biases.

## 8. Rekomendacje na przyszłość

1.  **Wdrożenie:** Model XGBoost (Tuned & Optimized) wraz z plikiem konfiguracyjnym progu może zostać wystawiony jako mikroserwis (np. REST API).

2.  **Monitoring:** Należy śledzić poziom _False Positive Rate_ na produkcji. Jeśli analitycy będą odrzucać zbyt wiele ostrzeżeń modelu, zalecana jest lekka korekta progu w górę.

3.  **Rozwój:** Warto rozważyć dodanie zmiennych makroekonomicznych (inflacja, PKB) do modelu, aby uodpornić go na zmiany rynkowe.

## 9. Skład grupy i podział zadań

- **Adam Łuczka:**
  - Pozyskanie danych ze źródła i ich unifikacja.
  - Eksploracyjna analiza danych (EDA): Analiza rozkładów zmiennych, badanie korelacji oraz identyfikacja braków danych.
  - Przetworzenie danych: Implementacja potoku przetwarzania (imputacja brakujących wartości, standaryzacja cech).
  - Redukcja wymiarowości: Zastosowanie analizy głównych składowych (PCA) do wizualizacji danych i selekcji cech.
  - Budowa i ewaluacja modelu bazowego.

- **Jakub Marciniak:**
  - Implementacja strategii radzenia z niezbalansowanymi danymi.
  - Budowa i optymalizacja zaawansowanych modeli zespołowych: Lasy losowy oraz XGBoost.
  - Konfiguracja środowiska Weights & Biases (WandB) do śledzenia eksperymentów i logowania wyników.
  - Analiza porównawcza modeli, badanie istotności cech i przygotowanie raportu końcowego.
