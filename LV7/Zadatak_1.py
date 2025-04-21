import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans

def generate_data(n_samples, flagc):
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                          centers=4,
                          cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                          random_state=random_state)
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    else:
        X = []
    return X

for mode in range(1, 6):
    X = generate_data(500, mode)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title(f'Način {mode} - Podatkovni primjeri')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    if mode in [1, 2]:
        K = 3
    elif mode == 3:
        K = 4
    else:
        K = 2

    kmeans = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                c='red', marker='X', s=200, edgecolor='black', label='Centri')
    plt.title(f'Način {mode} - KMeans klasteriranje (K={K})')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()

    plt.tight_layout()
    plt.show()
