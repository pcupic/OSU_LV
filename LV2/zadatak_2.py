import numpy as np
import matplotlib.pyplot as plt

filename = 'LV2/resources/data.csv'
data = np.loadtxt(filename, delimiter=",", skiprows=1)

num_people = data.shape[0]
print(f"Number of people measured: {num_people}")

gender = data[:, 0]  
height = data[:, 1] 
weight = data[:, 2] 

plt.figure()
plt.scatter(height, weight, alpha=0.5, label="All data")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight")
plt.legend()

plt.figure()
plt.scatter(height[::50], weight[::50], color='red', alpha=0.5, label="Every 50th person")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight (every 50th person)")
plt.legend()
plt.show()

print(f"Min height: {np.min(height):.2f} cm")
print(f"Max height: {np.max(height):.2f} cm")
print(f"Average height: {np.mean(height):.2f} cm")

male_indices = (gender == 1)
female_indices = (gender == 0)

print("\nMale height statistics:")
print(f"Min: {np.min(height[male_indices]):.2f} cm")
print(f"Max: {np.max(height[male_indices]):.2f} cm")
print(f"Average: {np.mean(height[male_indices]):.2f} cm")

print("\nFemale height statistics:")
print(f"Min: {np.min(height[female_indices]):.2f} cm")
print(f"Max: {np.max(height[female_indices]):.2f} cm")
print(f"Average: {np.mean(height[female_indices]):.2f} cm")
