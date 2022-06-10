import numpy as np
import cv2
from matplotlib import pyplot as plt
bgr = cv2.imread('input.png')
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(20,20))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_YUV2BGR)

# Load the image in color
#image_bgr = cv2.imread('input.png')
#image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
# create a CLAHE object (Arguments are optional).
#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#out = clahe.apply(image_bgr)
#image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
# Display the images side by side using cv2.hconcat
#out1 = cv2.hconcat([bgr,lab])
#cv2.imshow('a',out1)
#cv2.waitKey(0)
plt.imshow(bgr),plt.axis("off")
plt.show()
