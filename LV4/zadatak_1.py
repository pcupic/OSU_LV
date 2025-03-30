import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("LV4/resources/data_C02_emission.csv")

df.columns = df.columns.str.strip()

print("Nazivi stupaca:", df.columns.tolist())

numerical_features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption Comb (L/100km)']
target = 'CO2 Emissions (g/km)'

missing_columns = [col for col in numerical_features + [target] if col not in df.columns]
if missing_columns:
    raise KeyError(f"Sljedeći stupci nisu pronađeni u CSV datoteci: {missing_columns}")

X = df[numerical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.figure(figsize=(8, 5))
plt.scatter(X_train['Engine Size (L)'], y_train, color='blue', label='Train data', alpha=0.5)
plt.scatter(X_test['Engine Size (L)'], y_test, color='red', label='Test data', alpha=0.5)
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Ovisnost emisije CO2 o veličini motora')
plt.legend()
plt.show()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(X_train['Engine Size (L)'], bins=20, color='blue', alpha=0.7)
plt.title('Prije skaliranja')

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:, 0], bins=20, color='red', alpha=0.7)
plt.title('Nakon skaliranja')
plt.show()

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print(f"Koeficijenti: {model.coef_}")
print(f"Presjek s osi: {model.intercept_}")

y_pred = model.predict(X_test_scaled)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.xlabel('Stvarne vrijednosti CO2 emisije')
plt.ylabel('Predviđene vrijednosti CO2 emisije')
plt.title('Ovisnost stvarnih i predviđenih vrijednosti')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
