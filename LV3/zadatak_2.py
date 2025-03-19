import pandas as pd
import matplotlib.pyplot as plt

file = 'LV3/resources/data_C02_emission.csv'
df = pd.read_csv(file)

plt.figure(figsize=(10, 6))
plt.hist(df['CO2 Emissions (g/km)'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("CO2 Emissions (g/km)")
plt.ylabel("Frequency")
plt.title("Histogram of CO2 Emissions")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

fuel_colors = {'X': 'blue', 'Z': 'green', 'D': 'red', 'E': 'purple', 'N': 'orange'}
colors = df['Fuel Type'].map(fuel_colors)

plt.figure(figsize=(10, 6))
plt.scatter(df['Fuel Consumption City (L/100km)'], df['CO2 Emissions (g/km)'], c=colors, alpha=0.7, edgecolors='black')
plt.xlabel("Fuel Consumption City (L/100km)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Scatter Plot: Fuel Consumption vs CO2 Emissions")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

fuel_types = df['Fuel Type'].unique()
data = [df[df['Fuel Type'] == fuel]['Fuel Consumption Hwy (L/100km)'].dropna() for fuel in fuel_types]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=fuel_types)
plt.xlabel("Fuel Type")
plt.ylabel("Highway Fuel Consumption (L/100km)")
plt.title("Boxplot: Highway Fuel Consumption by Fuel Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

fuel_counts = df['Fuel Type'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(fuel_counts.index, fuel_counts.values, color='skyblue', edgecolor='black')
plt.xlabel("Fuel Type")
plt.ylabel("Number of Vehicles")
plt.title("Number of Vehicles by Fuel Type")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

cylinders_co2 = df.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()

plt.figure(figsize=(8, 6))
plt.bar(cylinders_co2.index, cylinders_co2.values, color='lightcoral', edgecolor='black')
plt.xlabel("Number of Cylinders")
plt.ylabel("Average CO2 Emissions (g/km)")
plt.title("Average CO2 Emissions by Number of Cylinders")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
