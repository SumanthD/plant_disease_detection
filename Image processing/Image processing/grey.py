import cv2 
import numpy as np 
from matplotlib import pyplot as plt
image = cv2.imread("leaf.jpg", cv2.IMREAD_GRAYSCALE)
plt.imshow(image, cmap="gray")
plt.axis("off") 
plt.show()