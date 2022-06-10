import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import feature,io
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt

img = io.imread('input5.jpg')

gray = color.rgb2gray(img)
image = img_as_ubyte(gray)

bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
inds = np.digitize(image, bins)

max_value = inds.max()+1
matrix = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)


gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#plt.imshow(gs,cmap='Greys_r')
#plt.imshow(img,cmap="Greys_r")
red_channel = img[:,:,0]
#plt.imshow(red_channel,cmap="Greys_r")
green_channel = img[:,:,1]
#plt.imshow(green_channel,cmap="Greys_r")
blue_channel = img[:,:,2]
#plt.imshow(blue_channel,cmap="Greys_r")
np.mean(blue_channel)
blue_channel[blue_channel == 255] = 0
green_channel[green_channel == 255] = 0
red_channel[red_channel == 255] = 0
red_mean = np.mean(red_channel)
print('Mean of Red',red_mean)
green_mean = np.mean(green_channel)
print('Mean of Green',green_mean)
blue_mean = np.mean(blue_channel)
print('Mean of Blue',blue_mean)
red_var = np.std(red_channel)
print('Std of Red',red_var)
blue_var = np.std(blue_channel)
print('Std of Blue',blue_var)
green_var = np.std(green_channel)
print('Std of Green',green_var)

ContrastStats = feature.greycoprops(matrix, 'contrast')
CorrelationStats = feature.greycoprops(matrix, 'correlation')
HomogeneityStats = feature.greycoprops(matrix, 'homogeneity')
EnergyStats = feature.greycoprops(matrix, 'energy')
ASMStats = feature.greycoprops(matrix, 'ASM')

print('Contrast = ',np.mean(ContrastStats))
print('Correlatin = ',np.mean(CorrelationStats))
print('Homogenity = ',np.mean(HomogeneityStats))
print('Energy = ',np.mean(EnergyStats))
print('Entropy = ',np.mean(ASMStats))
