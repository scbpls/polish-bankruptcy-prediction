# Temat: Predykcja upadłości polskich przedsiębiorstw na podstawie wskaźników finansowych

## 1. Skład grupy i podział zadań

- **Adam Łuczka:**
  - Pozyskanie danych ze źródła i ich unifikacja.
  - Eksploracyjna analiza danych (EDA) - analiza rozkładów zmiennych, badanie korelacji oraz identyfikacja braków danych.
  - Przetworzenie danych - implementacja potoku przetwarzania (imputacja brakujących wartości, standaryzacja cech).
  - Redukcja wymiarowości - zastosowanie analizy głównych składowych (PCA) do wizualizacji danych i selekcji cech.
  - Budowa i ewaluacja modelu bazowego.

- **Jakub Marciniak:**
  - Implementacja strategii radzenia z niezbalansowanymi danymi.
  - Budowa i optymalizacja zaawansowanych modeli zespołowych - lasy losowe oraz XGBoost.
  - Konfiguracja środowiska Weights & Biases (WandB) do śledzenia eksperymentów i logowania wyników.
  - Analiza porównawcza modeli, badanie istotności cech i przygotowanie raportu końcowego.

## 2. Opis prac

### Opis problemu

Głównym celem projektu jest opracowanie skutecznego modelu klasyfikacyjnego, zdolnego do prognozowania upadłości polskich przedsiębiorstw. Wykrywanie sygnałów ostrzegawczych o pogarszającej się sytuacji finansowej firmy jest kluczowym zagadnieniem w obszarze zarządzania ryzykiem kredytowym (dla banków) oraz inwestycyjnym.

Problem ten stanowi istotne wyzwanie algorytmiczne z dwóch powodów. Po pierwsze, mamy do czynienia z **silnym niezbalansowaniem klas** - liczba firm bankrutujących jest marginalna w porównaniu do firm stabilnych, co sprawia, że standardowe modele mają tendencję do ignorowania mniejszych klas. Po drugie, dane charakteryzują się wysoką wymiarowością, co rodzi ryzyko przetrenowania i trudności w interpretacji.

Projekt ma na celu stworzenie narzędzia wspomagającego decyzje, które zminimalizuje ryzyko błędnej oceny wiarygodności finansowej podmiotu.

### Zbiór danych

Do realizacji projektu wykorzystany zostanie zbiór danych **Polish Companies Bankruptcy**, dostępny w repozytorium UC Irvine Machine Learning Repository. Dane zostały zebrane z serwisu EMIS (Emerging Markets Information Service) i obejmują wskaźniki finansowe polskich firm.

- **Zakres czasowy:** Zbiór obejmuje dane z lat **2000-2013**. Firmy, które zbankrutowały, były analizowane w okresie 2000-2012, natomiast firmy, które nie upadły, w latach 2007-2013.
- **Struktura:** Dane podzielone są na 5 podzbiorów w zależności od okresu prognozy (od 1 roku do 5 lat przed ewentualnym bankructwem). Łącznie zbiór zawiera kilkadziesiąt tysięcy raportów finansowych (instancji).
- **Cechy:** Każda firma opisana jest przez 64 wskaźniki finansowe (m.in. zyskowność, zadłużenie, płynność).
- **Zmienna celu:** Binarna informacja o statusie bankructwa (0 - firma funkcjonuje, 1 - firma zbankrutowała).

### Lista metod

1.  **Inżynieria cech (feature engineering):** Zastosowanie metod imputacji (uzupełnianie braków danych medianą lub metodą k-NN) oraz skalowania (StandardScaler), co jest niezbędne dla modeli liniowych i PCA.
2.  **Uczenie nienadzorowane:** Wykorzystanie algorytmu **PCA (Principal Component Analysis)** w celu redukcji 64 wymiarów do mniejszej liczby głównych składowych oraz wizualizacji separacji klas w przestrzeni 2D/3D.
3.  **Uczenie nadzorowane:**
    - **Regresja logistyczna:** Jako model referencyjny, pozwalający ocenić liniowy podział danych.
    - **Lasy losowe:** Jako główny model predykcyjny, odporny na szum i zdolny do modelowania nieliniowych zależności. Wykorzystany zostanie również do oceny ważności poszczególnych wskaźników finansowych.
    - **Extreme Gradient Boosting (XGBoost):** Jako nowoczesna metoda dla danych tabelarycznych.

### Opis miar do oceny jakości

Ze względu na specyfikę problemu (dane niezbalansowane), standardowa metryka _dokładność_ (accuracy) nie będzie brana pod uwagę jako główne kryterium oceny. Ocena modeli oparta zostanie na:

- **Czułość (recall):** Kluczowa miara w projekcie. Zależy nam na maksymalizacji wykrywalności bankrutów (minimalizacja błędu False Negative), nawet kosztem częstszych fałszywych alarmów.
- **Pole pod krzywą ROC (ROC-AUC):** Miara globalnej zdolności modelu do rozróżniania klas przy różnych progach decyzyjnych.
- **Miara F1 (F1 Score):** Średnia harmoniczna precyzji i czułości.
- **Macierz pomyłek (confusion matrix):** Do wizualizacji błędów I i II rodzaju.

Wszystkie eksperymenty, wraz z wykresami uczenia i wynikami metryk, będą rejestrowane na platformie **Weights & Biases (WandB)**.
