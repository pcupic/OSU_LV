import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

img = Image.imread("LV7/resources/imgs/test_6.jpg")

plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.axis('off')
plt.tight_layout()
plt.show()

img = img.astype(np.float64) / 255

w, h, d = img.shape
img_array = np.reshape(img, (w * h, d))

unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

Ks = [2, 5, 8, 16]
fig, axs = plt.subplots(1, len(Ks), figsize=(4 * len(Ks), 5))

for idx, K in enumerate(Ks):
    print(f"\nK = {K}")

    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
    labels = kmeans.fit_predict(img_array)
    centers = kmeans.cluster_centers_

    quantized_img_array = centers[labels]
    quantized_img = quantized_img_array.reshape((w, h, d))

    axs[idx].imshow(quantized_img)
    axs[idx].set_title(f"K = {K}")
    axs[idx].axis('off')

plt.suptitle("Kvantizacija boja za različite vrijednosti K", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

inertias = []
K_range = range(1, 11)
for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
    kmeans.fit(img_array)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertias, marker='o')
plt.title("Lakat metoda: J ovisno o K")
plt.xlabel("Broj grupa K")
plt.ylabel("Vrijednost J (inertia)")
plt.grid(True)
plt.show()

K = 5
kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
labels = kmeans.fit_predict(img_array)
labels_img = labels.reshape((w, h))

fig, axs = plt.subplots(1, K, figsize=(4 * K, 5))

for i in range(K):
    binary_mask = (labels_img == i).astype(float)
    binary_rgb = np.stack([binary_mask] * 3, axis=-1)

    axs[i].imshow(binary_rgb, cmap="gray")
    axs[i].set_title(f"Grupa {i+1}")
    axs[i].axis('off')

plt.suptitle("Binarne slike za K = 5", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
