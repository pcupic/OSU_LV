import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

# Broj klasa
num_classes = 10
input_shape = (28, 28, 1)

# Učitaj spremljeni model
model = keras.models.load_model("C:/Users/hp/Desktop/mnist_cnn_model.h5")
print("Model učitan.")

# Učitaj MNIST skup podataka
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Skaliranje slika
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# Dobij predikcije
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)

# Pronađi indekse loše klasificiranih slika
wrong_indices = np.where(y_pred_classes != y_test)[0]
print(f"Broj lose klasificiranih primjera: {len(wrong_indices)}")

# Prikaz nekoliko loše klasificiranih slika
plt.figure(figsize=(12, 6))
for i in range(10):
    idx = wrong_indices[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"True: {y_test[idx]}, Pred: {y_pred_classes[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.suptitle("Loše klasificirane slike", fontsize=16, y=1.05)
plt.show()
