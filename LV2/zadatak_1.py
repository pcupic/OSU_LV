import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 3, 3, 2, 1])  
y = np.array([1, 1, 2, 2, 1])

plt.plot(x, y, color='red', linewidth=3, linestyle='--', marker='o', markerfacecolor='blue', markersize=8)

plt.title("Example Graph", fontsize=14)
plt.xlabel("X-axis", fontsize=12)
plt.ylabel("Y-axis", fontsize=12)

plt.grid(True)
plt.show()
