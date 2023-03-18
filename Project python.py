
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Importer les données historiques de la bourse du Liban
dl = pd.read_csv(r'C:\Users\gaga-\Documents\4A\Business Analytics (Python & Excel VBA)\Python\Paris-G2-Bousselmi-Elegoet\BLOM.csv')

# Charger le fichier CSV dans une DataFrame pandas
df = pd.read_csv(dl, encoding='iso-8859-1')

# Afficher les premières lignes de la DataFrame
print(df.head())

# Diviser les données en ensemble d'entraînement et de test
train_data = data.iloc[:-100, :]
test_data = data.iloc[-100:, :]

# Créer une fonction pour préparer les données pour la régression linéaire
def prepare_data(df):
    X = np.array(df['Date'].index).reshape(-1, 1)
    y = np.array(df['Price'])
    return X, y

# Préparer les données d'entraînement et de test
X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

# Créer et entraîner un modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédire les valeurs de la bourse pour l'ensemble de test
y_pred = model.predict(X_test)

# Afficher les valeurs prédites et les valeurs réelles
print(pd.DataFrame({'Predicted': y_pred, 'Actual': y_test}))

# Tracer le modèle de régression linéaire et les valeurs réelles
plt.plot(X_train, y_train, 'o', color='blue')
plt.plot(X_test, y_test, 'o', color='green')
plt.plot(X_test, y_pred, '-', color='red')
plt.xlabel('Date')
plt.ylabel('Prix (USD)')
plt.legend(['Training Data', 'Test Data', 'Predicted'])
plt.show()
