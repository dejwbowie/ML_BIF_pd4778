import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1) Załadowanie danych
df = pd.read_csv('dane_projekt1.csv')

# 2) Eksploracja danych
print(df.head())

print(df.describe())
# srednia wszystkich zmiennych numerycznych oscyluje w podobnych zakresach, podobnie, jak wartosci minimalne i maksymalne,
# dlatego tez nie ma potrzeby skalowania danych

print(df.isna().sum())
# brak brakujacych wartosci

# Sprawdzenie dostępnych klas i liczebności próbek w grupach
print(df.groupby('Gene_Function').size())

# 3) Przetworzenie danych w celu trenowania modelu
X = df.drop(columns = ['Gene_ID', 'Gene_Function'])
y = df['Gene_Function']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y, random_state = 42)

# 4) Budowa modelu
# Wybrano Las Losowy ze względu na dobre przystosowanie do zbiorów danych RNA-seq - radzi sobie z overfittingiem,
# który można obserwować w zbiorach o wysokiej wymiarowości, a także wdrożenie treningu wielu drzew na różnych próbkach danych,
# co pozwala lepiej wykorzystać niewielka liczbę próbek
model = RandomForestClassifier(n_estimators = 150, max_depth = 20, bootstrap = True,
                               class_weight = 'balanced', min_samples_leaf = 6)

model.fit(X_train, y_train)

# Predykcja
y_pred = model.predict(X_test)

# 5) Ewaluacja modelu
print(classification_report(y_test, y_pred))

# Model nie osiaga zbyt dobrych wynikow, wartosci accuracy oscyluja między 15% a 33%, w tym spośród niektórych kategorii
# żadne próbki nie zostały prawidłowo zakwalifikowane, osiagajac wartosc precision oraz recall = 0;
# po części wynika to na pewno z roznej liczebności klas, przez co preferowane moga być te o większej liczebności (dla klasy 
# receptor precision wyniosła 50%), oraz ogólnie niskiej liczebności zbioru. 