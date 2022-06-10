import cv2 
import numpy as np 
from matplotlib import pyplot as plt
image = cv2.imread("leaf.jpg", cv2.IMREAD_GRAYSCALE)
image_50x50 = cv2.resize(image, (50, 50))
plt.imshow(image_50x50, cmap="gray")
plt.axis("off") 
plt.show()
