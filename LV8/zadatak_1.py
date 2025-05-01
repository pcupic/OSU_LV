import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Učitavanje podataka
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Prikaz osnovnih informacija
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

# Prikaz nekoliko slika iz train skupa
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Preoblikovanje: slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)

print("x_train shape:", x_train_s.shape)
print(x_train_s.shape[0], "train samples")
print(x_test_s.shape[0], "test samples")

# Pretvaranje labela u kategorije
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)

# Kreiranje modela
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Prikaz strukture modela
model.summary()

# Kompilacija modela
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Treniranje modela
history = model.fit(x_train_s, y_train_s,
                    batch_size=128,
                    epochs=5,
                    validation_split=0.1,
                    verbose=2)

# Evaluacija modela na test skupu
test_loss, test_acc = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Matrica zabune
y_pred = model.predict(x_test_s)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

# Prikaz matrice zabune
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Spremanje modela
model.save("C:/Users/hp/Desktop/mnist_cnn_model.h5")
print("Model je spremljen kao 'mnist_cnn_model.h5'")
