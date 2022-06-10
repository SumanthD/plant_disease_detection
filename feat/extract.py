from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import cv2
import os


image1read = cv2.imread('image1.jpg')
(h1, w1) = image1read.shape[:2]
image2read = cv2.imread('image3.jpg')
(h2, w2) = image2read.shape[:2]
image3read = cv2.imread('image4.jpg')
(h3, w3) = image3read.shape[:2]



image1 = cv2.cvtColor(image1read, cv2.COLOR_BGR2LAB)
image2 = cv2.cvtColor(image2read, cv2.COLOR_BGR2LAB)
image3 = cv2.cvtColor(image3read, cv2.COLOR_BGR2LAB)


image1 = image1.reshape((image1.shape[0] * image1.shape[1], 3))
image2 = image2.reshape((image2.shape[0] * image2.shape[1], 3))
image3 = image3.reshape((image3.shape[0] * image3.shape[1], 3))


# clt = MiniBatchKMeans(n_clusters = 16)
clt = KMeans(n_clusters = 8)

labels1 = clt.fit_predict(image1)
quant1 = clt.cluster_centers_.astype("uint8")[labels1]

print ("1st done")
labels2 = clt.fit_predict(image2)
quant2 = clt.cluster_centers_.astype("uint8")[labels2]

print ("2nd done")
labels3 = clt.fit_predict(image3)
quant3 = clt.cluster_centers_.astype("uint8")[labels3]

print ("3rd done")


#reshape the feature vectors to images
quant1 = quant1.reshape((h1, w1, 3))
image1 = image1.reshape((h1, w1, 3))

quant2 = quant2.reshape((h2, w2, 3))
image2 = image2.reshape((h2, w2, 3))

quant3 = quant3.reshape((h3, w3, 3))
image3 = image3.reshape((h3, w3, 3))

# convert from L*a*b* to RGB
quant1 = cv2.cvtColor(quant1, cv2.COLOR_LAB2BGR)
image1 = cv2.cvtColor(image1, cv2.COLOR_LAB2BGR)

quant2 = cv2.cvtColor(quant2, cv2.COLOR_LAB2BGR)
image2 = cv2.cvtColor(image2, cv2.COLOR_LAB2BGR)

quant3 = cv2.cvtColor(quant3, cv2.COLOR_LAB2BGR)
image3 = cv2.cvtColor(image3, cv2.COLOR_LAB2BGR)

path = 'clusteredImages'
os.mkdir(path)

    # disable only the cluster number 2 (turn the pixel into black)
masked_image = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=0
#masked_image[labels == cluster] = [0, 0, 0]
masked_image[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image = masked_image.reshape(image1.shape)

masked_image = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=0
#masked_image[labels == cluster] = [0, 0, 0]
masked_image[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image = masked_image.reshape(image1.shape)

masked_image1 = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image1 = masked_image1.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=1
#masked_image[labels == cluster] = [0, 0, 0]
masked_image1[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image1 = masked_image1.reshape(image1.shape)

masked_image2 = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image2 = masked_image2.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=2
#masked_image[labels == cluster] = [0, 0, 0]
masked_image2[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image2 = masked_image2.reshape(image1.shape)

masked_image3 = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image3 = masked_image3.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=3
#masked_image[labels == cluster] = [0, 0, 0]
masked_image3[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image3 = masked_image3.reshape(image1.shape)
# show the image
 #   os.chdir(cwd+r'\output_1')
  #  cv2.imwrite(i,masked_image)



masked_image4 = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image4 = masked_image4.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=4
#masked_image[labels == cluster] = [0, 0, 0]
masked_image4[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image4 = masked_image4.reshape(image1.shape)


masked_image5 = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image5 = masked_image5.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=5
#masked_image[labels == cluster] = [0, 0, 0]
masked_image5[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image5 = masked_image5.reshape(image1.shape)


masked_image6 = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image6 = masked_image6.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=6
#masked_image[labels == cluster] = [0, 0, 0]
masked_image6[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image6 = masked_image6.reshape(image1.shape)


masked_image7 = np.copy(image1)
# convert to the shape of a vector of pixel values
masked_image7 = masked_image7.reshape((-1, 3))
# color (i.e cluster) to disable

cluster=7
#masked_image[labels == cluster] = [0, 0, 0]
masked_image7[labels1 != cluster] = [255, 255, 255]
# convert back to original shape

masked_image7 = masked_image7.reshape(image1.shape)

cv2.imwrite(os.path.join(path , 'image1reduced.jpg'), masked_image)
cv2.imwrite(os.path.join(path , 'image4reduced.jpg'), masked_image1)
cv2.imwrite(os.path.join(path , 'image5reduced.jpg'), masked_image2)
cv2.imwrite(os.path.join(path , 'image6reduced.jpg'), masked_image3)
cv2.imwrite(os.path.join(path , 'image7reduced.jpg'), masked_image4)
cv2.imwrite(os.path.join(path , 'image8reduced.jpg'), masked_image5)
cv2.imwrite(os.path.join(path , 'image9reduced.jpg'), masked_image6)
cv2.imwrite(os.path.join(path , 'image10reduced.jpg'), masked_image7)
cv2.imwrite(os.path.join(path , 'image2reduced.jpg'), quant1)
cv2.imwrite(os.path.join(path , 'image3reduced.jpg'), quant3)
cv2.waitKey(0)
cv2.destroyAllWindows()







