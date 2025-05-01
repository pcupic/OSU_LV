import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt

model = keras.models.load_model("C:/Users/hp/Desktop/mnist_cnn_model.h5")
print("Model učitan.")

img_path = "LV8/resources/test.png"
img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")

plt.imshow(img, cmap="gray")
plt.title("Originalna slika")
plt.axis("off")
plt.show()

img_array = image.img_to_array(img)
img_array = img_array.astype("float32") / 255
img_array = np.expand_dims(img_array, axis=-1)

pred = model.predict(np.expand_dims(img_array, axis=0))
pred_class = np.argmax(pred, axis=1)[0]

print(f"Predviđena oznaka za sliku je: {pred_class}")
