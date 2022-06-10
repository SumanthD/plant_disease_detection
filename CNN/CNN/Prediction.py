import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils

bg = None

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,basewidth), Image.ANTIALIAS)
    img.save(imageName)


    # compute weighted average, accumulate it and update the background
    #cv2.accumulateWeighted(image, bg, aWeight)


def main():

    resizeImage('scab.JPG')
    predictedClass, confidence = getPredictedClass();
    showStatistics(predictedClass, confidence)

def getPredictedClass():
    # Predict
    image = cv2.imread('scab.JPG')
    #ray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([image.reshape(100, 100, 3)])
    print(prediction)
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] + prediction[0][5] + prediction[0][6] + prediction[0][7]))

def showStatistics(predictedClass, confidence):

    textImage = np.zeros((300,512,3), np.uint8)
    className = ""
    if predictedClass == 0:
        className = "Apple scab"
        print(className)
    elif predictedClass == 1:
        className = "Bacterial Spot"
        print(className)
    elif predictedClass == 2:
        className = "Black Rot"
        print(className)
    elif predictedClass == 3:
        className = "Cedar Apple Rust"
        print(className)
    elif predictedClass == 4:
        className = "Early Blight"
        print(className)
    elif predictedClass == 5:
        className = "Late Blight"
        print(className)
    elif predictedClass == 6:
        className = "Mosaic Virus"
        print(className)
    elif predictedClass == 7:
        className = "Powdery Mildew"
    cv2.putText(textImage,"Disease : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    1,
    (255, 255, 255),
    2)

    #cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    #(30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2)
    cv2.imshow("Statistics", textImage)




# Model defined
tf.reset_default_graph()
convnet=input_data(shape=[None,100,100,3],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,8,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Load Saved Model
model.load("GestureRecogModel.tfl")

main()
