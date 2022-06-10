import cv2 
import numpy as np 
from matplotlib import pyplot as plt
image = cv2.imread("leaf.jpg", cv2.IMREAD_GRAYSCALE)
image_cropped = image[:,:128]
plt.imshow(image_cropped, cmap="gray"), plt.axis("off")
plt.show()
