"""
TP 1 - CNN : classification d'images de vêtements (Fashion-MNIST)

Objectif : construire un réseau de neurones convolutif capable de reconnaître
10 catégories de vêtements à partir d'images 28x28 en niveaux de gris.
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers

keras.utils.set_random_seed(42)  # résultats reproductibles

# Étape 1 : chargement des données (téléchargement automatique)

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

noms_classes = ["T-shirt", "Pantalon", "Pull", "Robe", "Manteau",
                "Sandale", "Chemise", "Basket", "Sac", "Bottine"]

print("Images d'entraînement :", x_train.shape)  # (60000, 28, 28)
print("Images de test        :", x_test.shape)   # (10000, 28, 28)

# Étape 2 : préparation des images

# Pixels ramenés entre 0 et 1 (au lieu de 0 à 255)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Ajout de la dimension "canal" : (60000, 28, 28) -> (60000, 28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Étape 3 : construction du modèle CNN

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),   # détecte des motifs simples
    layers.MaxPooling2D((2, 2)),                    # réduit la taille de moitié
    layers.Conv2D(64, (3, 3), activation="relu"),   # détecte des motifs plus complexes
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),                               # met les données à plat
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),         # 10 probabilités (une par classe)
])

model.summary()

# Étape 4 : compilation et entraînement

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_split=0.1)

# Étape 5 : évaluation sur le jeu de test

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nPrécision sur le jeu de test : {test_acc:.4f}")

# Étape 6 : courbes d'apprentissage

plt.plot(history.history["accuracy"], label="entraînement")
plt.plot(history.history["val_accuracy"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Précision")
plt.title("TP1 - Courbes d'apprentissage")
plt.legend()
plt.savefig("tp1_courbes.png")
plt.show()

# Étape 7 : prédiction sur quelques images du jeu de test

probas = model.predict(x_test[:5])
classes_predites = np.argmax(probas, axis=1)

plt.figure(figsize=(10, 3))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Prédit : {noms_classes[classes_predites[i]]}\nRéel : {noms_classes[y_test[i]]}",
              fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.savefig("tp1_predictions.png")
plt.show()
