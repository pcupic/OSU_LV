import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', label='Podaci za učenje')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr', marker='x', label='Podaci za testiranje')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podaci za učenje i testiranje')
plt.show()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

theta0 = model.intercept_[0]
theta1 = model.coef_[0][0]
theta2 = model.coef_[0][1]

print(f"Parametri modela: theta0 = {theta0}, theta1 = {theta1}, theta2 = {theta2}")

x1 = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
x2 = (-theta0 - theta1 * x1) / theta2

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', label='Podaci za učenje')
plt.plot(x1, x2, color='black', label='Granica odluke')
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Granica odluke')
plt.show()

y_pred = model.predict(X_test)

matrix_zabune = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(matrix_zabune)

tocnost = accuracy_score(y_test, y_pred)
preciznost = precision_score(y_test, y_pred)
odziv = recall_score(y_test, y_pred)

print(f"Tocnost: {tocnost}")
print(f"Preciznost: {preciznost}")
print(f"Odziv: {odziv}")

plt.figure(figsize=(8, 6))
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        plt.scatter(X_test[i, 0], X_test[i, 1], color='green')
    else:
        plt.scatter(X_test[i, 0], X_test[i, 1], color='black')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Testni podaci s označenim klasifikacijama')
plt.show()
