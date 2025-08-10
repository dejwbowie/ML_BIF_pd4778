import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('dane_projekt1.csv')

print(df.describe())

X = df.drop(columns = 'Gene_Function')
y = df['Gene_Function']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size = 0.15, stratify = y, random_state = 42)

X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size = 0.2, stratify = y_temp, random_state = 42)