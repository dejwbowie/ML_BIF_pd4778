import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

data = {
    'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8],
    'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9],
    'TF_motifs': [2, 6, 1, 4, 5, 2, 6, 3],
    'KRAS':   [1.2, 7.1, 0.9, 6.8, 1.5, 5.5, 1.0, 6.3],
    'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0]  
}

df = pd.DataFrame(data)

X = df[['TP53_expr', 'BRCA1_expr', 'TF_motifs', 'KRAS']]
y = df['Cancer_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Model poradził sobie idealnie na zbiorze testowym, przewidział raka u pacjentów, u których faktycznie
# występuje, dlatego też wszystkie metryki osignęły maksymalne wartości = 1.0