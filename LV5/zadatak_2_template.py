import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

labels = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], edgecolor='k',
                    label=labels[cl])

df = pd.read_csv("LV5/resources/penguins.csv")

df.drop(columns=['sex'], inplace=True)
df.dropna(axis=0, inplace=True)
df['species'].replace({'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}, inplace=True)

X = df[['bill_length_mm', 'flipper_length_mm']].to_numpy()
y = df['species'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

train_classes, train_counts = np.unique(y_train, return_counts=True)
test_classes, test_counts = np.unique(y_test, return_counts=True)

plt.figure(figsize=(8, 6))
plt.bar(train_classes - 0.2, train_counts, width=0.4, label='Training Set')
plt.bar(test_classes + 0.2, test_counts, width=0.4, label='Testing Set')
plt.xticks([0, 1, 2], ['Adelie', 'Chinstrap', 'Gentoo'])
plt.xlabel('Penguin Species')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training and Testing Sets')
plt.legend()
plt.show()

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model Parameters:")
print(f"Intercepts: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
print("Difference from binary classification: The model now has separate intercepts and coefficients for each class due to multi-class classification.")

plot_decision_regions(X_train, y_train.ravel(), classifier=model)
plt.xlabel('Bill Length (mm)')
plt.ylabel('Flipper Length (mm)')
plt.title('Decision Regions for Training Data')
plt.legend()
plt.show()

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test.ravel(), y_pred)
print("Confusion Matrix:")
print(conf_matrix)

accuracy = accuracy_score(y_test.ravel(), y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test.ravel(), y_pred))

if 'body_mass_g' in df.columns:
    X_full = df[['bill_length_mm', 'flipper_length_mm', 'body_mass_g']].to_numpy()
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, y, test_size=0.2,
                                                                           random_state=123)
    model_full = LogisticRegression(max_iter=1000)
    model_full.fit(X_train_full, y_train_full)
    
    y_pred_full = model_full.predict(X_test_full)
    accuracy_full = accuracy_score(y_test_full.ravel(), y_pred_full)
    
    print(f"Accuracy with additional feature (body_mass_g): {accuracy_full}")
