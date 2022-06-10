import cv2 
import numpy as np 
from matplotlib import pyplot as plt
image = cv2.imread("leaf.jpg", cv2.IMREAD_GRAYSCALE)
image_blurry = cv2.blur(image, (5,5))
plt.imshow(image_blurry, cmap="gray"), plt.axis("off") 
plt.show()
