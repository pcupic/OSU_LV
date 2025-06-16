import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("LV2/resources/road.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

bright_image = np.clip(image + 50, 0, 255).astype(np.uint8)

height, width, _ = image.shape
second_quarter = image[:, width//4:width//2]

rotated_image = np.rot90(image, k=-1)

mirrored_image = np.fliplr(image)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
axes[0].imshow(bright_image)
axes[0].set_title("Brightened")
axes[1].imshow(second_quarter)
axes[1].set_title("Second Quarter")
axes[2].imshow(rotated_image)
axes[2].set_title("Rotated 90°")
axes[3].imshow(mirrored_image)
axes[3].set_title("Mirrored")

for ax in axes:
    ax.axis("off")

plt.show()
