"""
TP 3 - NLP : classification automatique de commentaires en français
(positif / négatif)

Dataset : Allociné (critiques de films en français), via Hugging Face.
Labels : 0 = négatif, 1 = positif.
"""

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import keras
from keras import layers

keras.utils.set_random_seed(42)  # résultats reproductibles

# Étape 1 : chargement du dataset (téléchargé au premier lancement)

try:
    dataset = load_dataset("allocine")
except Exception:
    dataset = load_dataset("tblard/allocine")

# On garde un sous-ensemble pour que l'entraînement reste rapide
train = dataset["train"].shuffle(seed=42).select(range(20000))
test = dataset["test"].shuffle(seed=42).select(range(5000))

textes_train = np.array(train["review"])
y_train = np.array(train["label"])
textes_test = np.array(test["review"])
y_test = np.array(test["label"])

print("Commentaires d'entraînement :", len(textes_train))
print("Commentaires de test        :", len(textes_test))
print("Exemple :", textes_train[0][:100], "... -> label :", y_train[0])

# Étape 2 : transformation des textes en nombres

nb_mots = 10000       # on garde les 10 000 mots les plus fréquents
longueur_max = 200    # chaque commentaire fait exactement 200 mots

# TextVectorization met en minuscules, enlève la ponctuation, découpe en mots,
# puis remplace chaque mot par un numéro. Il complète/coupe à 200 mots.
vectorizer = layers.TextVectorization(max_tokens=nb_mots,
                                      output_sequence_length=longueur_max)
vectorizer.adapt(textes_train)   # apprend le vocabulaire sur l'entraînement uniquement

x_train = vectorizer(textes_train).numpy()
x_test = vectorizer(textes_test).numpy()
print("Forme de x_train :", x_train.shape)  # (20000, 200)

# Étape 3 : construction du modèle

model = keras.Sequential([
    layers.Input(shape=(longueur_max,)),
    layers.Embedding(nb_mots, 16),            # chaque mot -> vecteur de 16 nombres
    layers.GlobalAveragePooling1D(),          # moyenne des vecteurs du commentaire
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),    # score entre 0 (négatif) et 1 (positif)
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# Étape 4 : entraînement

history = model.fit(x_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2)

# Étape 5 : évaluation sur le jeu de test

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nPrécision sur le jeu de test : {test_acc:.4f}")

plt.plot(history.history["accuracy"], label="entraînement")
plt.plot(history.history["val_accuracy"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Précision")
plt.title("TP3 - Courbes d'apprentissage")
plt.legend()
plt.savefig("tp3_courbes.png")
plt.show()

# Étape 6 : tester avec des commentaires écrits par nous

def predire(commentaire):
    sequence = vectorizer([commentaire])          # même transformation que l'entraînement
    score = model.predict(sequence)[0][0]
    return ("positif" if score > 0.5 else "négatif"), float(score)

exemples = [
    "Un film magnifique, les acteurs sont excellents et l'histoire est captivante.",
    "Quel ennui, un scénario nul et des acteurs mauvais, une vraie perte de temps.",
]
for texte in exemples:
    sentiment, score = predire(texte)
    print(f"{sentiment} ({score:.2f}) -> {texte}")
