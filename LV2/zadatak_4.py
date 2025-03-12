import numpy as np
import cv2
import matplotlib.pyplot as plt

black_square = np.zeros((50, 50), dtype=np.uint8)  
white_square = np.ones((50, 50), dtype=np.uint8) * 255  

row1 = np.hstack((black_square, white_square))
row2 = np.hstack((white_square, black_square))
checkerboard = np.vstack((row1, row2))

plt.imshow(checkerboard, cmap="gray")
plt.axis("off")
plt.title("Checkerboard Pattern")
plt.show()

