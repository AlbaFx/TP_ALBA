"""
TP 2 - RNN : prédiction d'une série temporelle

Objectif : construire un RNN simple qui apprend une série numérique et prédit
la valeur suivante à partir des valeurs précédentes.

Dataset : série générée par code (sinusoïde + bruit aléatoire).
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers

keras.utils.set_random_seed(42)  # résultats reproductibles

# Partie A - Étape 1 : génération des données

temps = np.arange(1000)
serie = np.sin(0.1 * temps) + np.random.normal(0, 0.05, 1000)  # sinus + bruit

plt.plot(serie[:200])
plt.title("TP2 - Série temporelle générée (200 premiers points)")
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.savefig("tp2_serie.png")
plt.show()

# Étape 2 : création des fenêtres (entrées / cibles)

taille_fenetre = 20   # on regarde les 20 valeurs passées
X, y = [], []

for i in range(len(serie) - taille_fenetre):
    X.append(serie[i : i + taille_fenetre])   # 20 valeurs consécutives
    y.append(serie[i + taille_fenetre])       # la valeur qui suit (à prédire)

# Forme attendue par un RNN : (nb exemples, nb pas de temps, nb variables)
X = np.array(X).reshape(-1, taille_fenetre, 1)
y = np.array(y)
print("Forme de X :", X.shape)  # (980, 20, 1)
print("Forme de y :", y.shape)  # (980,)

# Étape 3 : séparation entraînement / test (dans l'ordre chronologique)

coupure = int(0.8 * len(X))
X_train, X_test = X[:coupure], X[coupure:]
y_train, y_test = y[:coupure], y[coupure:]

# Étape 4 : construction et entraînement du RNN

model = keras.Sequential([
    layers.Input(shape=(taille_fenetre, 1)),
    layers.SimpleRNN(32),   # lit la fenêtre valeur par valeur
    layers.Dense(1),        # sortie : une seule valeur (la valeur suivante)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.1)


# Étape 5 : évaluation

test_mse = model.evaluate(X_test, y_test)
rmse = np.sqrt(test_mse)
print(f"\nRMSE sur le jeu de test : {rmse:.4f}")

y_pred = model.predict(X_test).flatten()

# Étape 6 : visualisation des prédictions

plt.plot(y_test[:100], label="valeurs réelles")
plt.plot(y_pred[:100], label="prédictions")
plt.title("TP2 - Prédiction de la valeur suivante")
plt.xlabel("Temps")
plt.ylabel("Valeur")
plt.legend()
plt.savefig("tp2_predictions.png")
plt.show()

# Étape 7 : prédire la toute prochaine valeur après la fin de la série

derniere_fenetre = serie[-taille_fenetre:].reshape(1, taille_fenetre, 1)
prochaine_valeur = model.predict(derniere_fenetre)[0][0]
print("Prochaine valeur prédite :", prochaine_valeur)
