import cv2
import numpy as np
from matplotlib import pyplot as plt
# Load image
image = cv2.imread("images.jpeg", cv2.IMREAD_GRAYSCALE)
# Enhance image
image_enhanced = cv2.equalizeHist(image)
# Show image
plt.imshow(image_enhanced,cmap="gray")
plt.axis("off")
plt.show()
