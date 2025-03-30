import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error

df = pd.read_csv("LV4/resources/data_C02_emission.csv")
df.columns = df.columns.str.strip()

numerical_features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption Comb (L/100km)']
categorical_feature = 'Fuel Type'
target = 'CO2 Emissions (g/km)'

ohe = OneHotEncoder(drop='first', sparse_output=False)
categorical_encoded = ohe.fit_transform(df[[categorical_feature]])
categorical_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out([categorical_feature]))

X = pd.concat([df[numerical_features], categorical_df], axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Maksimalna pogreska: {max_err:.2f} g/km")

test_indices = X_test.index
max_error_index = test_indices[np.argmax(np.abs(y_test - y_pred))]
vehicle_info = df.loc[max_error_index, ['Make', 'Model', 'Vehicle Class']]
print(f"Vozilo s maksimalnom pogreskom: {vehicle_info.to_dict()}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.xlabel('Stvarne vrijednosti CO2 emisije')
plt.ylabel('Predviđene vrijednosti CO2 emisije')
plt.title('Ovisnost stvarnih i predviđenih vrijednosti')
plt.show()
