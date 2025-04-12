import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("LV6/resources/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_train_knn = KNN_model.predict(X_train_n)
y_test_knn = KNN_model.predict(X_test_n)

print("\nKNN klasifikator (K=5):")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn)))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("KNN K=5, Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn)))
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


for k in [1, 100]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_n, y_train)

    y_test_k = knn.predict(X_test_n)

    print(f"\nKNN klasifikator (K={k}):")
    print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_k)))

    plot_decision_regions(X_train_n, y_train, classifier=knn)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(f"KNN K={k}, Granica odluke")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


param_grid = {'n_neighbors': np.arange(1, 50)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_knn.fit(X_train_n, y_train)

print("\nNajbolji K (preko unakrsne validacije):", grid_knn.best_params_)
print("Tocnost na test skupu: " + "{:0.3f}".format(grid_knn.score(X_test_n, y_test)))

plot_decision_regions(X_train_n, y_train, classifier=grid_knn.best_estimator_)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("Optimalni KNN, K=" + str(grid_knn.best_params_['n_neighbors']))
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


for C_val, gamma_val in [(1, 0.1), (100, 0.1), (1, 1), (100, 1)]:
    svm_rbf = svm.SVC(kernel='rbf', C=C_val, gamma=gamma_val)
    svm_rbf.fit(X_train_n, y_train)

    y_test_svm = svm_rbf.predict(X_test_n)
    acc = accuracy_score(y_test, y_test_svm)

    print(f"\nSVM (RBF), C={C_val}, gamma={gamma_val}")
    print("Tocnost test: " + "{:0.3f}".format(acc))

    plot_decision_regions(X_train_n, y_train, classifier=svm_rbf)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(f"SVM RBF, C={C_val}, γ={gamma_val}")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


for kernel in ['linear', 'poly', 'sigmoid']:
    svm_model = svm.SVC(kernel=kernel, C=1)
    svm_model.fit(X_train_n, y_train)

    print(f"\nSVM kernel: {kernel}")
    print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, svm_model.predict(X_test_n))))

    plot_decision_regions(X_train_n, y_train, classifier=svm_model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(f"SVM kernel: {kernel}")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

grid_svm = GridSearchCV(svm.SVC(), param_grid_svm, cv=5)
grid_svm.fit(X_train_n, y_train)

print("\nNajbolji SVM parametri (preko unakrsne validacije):", grid_svm.best_params_)
print("Tocnost na test skupu: " + "{:0.3f}".format(grid_svm.score(X_test_n, y_test)))

plot_decision_regions(X_train_n, y_train, classifier=grid_svm.best_estimator_)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("Optimalni SVM (RBF)")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
