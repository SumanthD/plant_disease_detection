import cv2
import numpy
import matplotlib.pyplot as plt
import os
import subprocess
import numpy as np
subprocess.call("mkdir output",shell=True)
cwd=os.getcwd()
os.chdir(cwd+r'\input');
a=[i for i in os.listdir() if ".jpg" or ".jpeg" or ".png" in i]
for i in a:
    os.chdir(cwd+r'\input')
    bgr = cv2.imread(i)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    #print(pixel_values.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 7
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

# flatten the labels array
    labels = labels.flatten()
# convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
# reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
# convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable

    cluster=2
#masked_image[labels == cluster] = [0, 0, 0]
    masked_image[labels != cluster] = [255, 255, 255]
# convert back to original shape

    masked_image = masked_image.reshape(image.shape)
# show the image
    os.chdir(cwd+r'\output_1')
    cv2.imwrite(i,masked_image)

