import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import socket
import cv2
import numpy
import time
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte
from sklearn import preprocessing,svm
import pickle
import tensorflow
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from PIL import Image
import imutils
import threading



#global variables
trained_model=open("trained_model_5_poly.pickle","rb")
output_array=['Apple_scab','Bacterial_spot','Black_rot','Early_blight','Late_blight','Mosaic_virus','Powdery_Mildew','Rust']


# Model defined
tensorflow.reset_default_graph()
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
clf=pickle.load(trained_model)
model.load("GestureRecogModel.tfl")
count=0

#kivy language
Builder.load_string('''
<main_screen>:
    BoxLayout:
        orientation:"vertical"
        BoxLayout:
            orientation:"horizontal"
            size_hint:1,0.9
            BoxLayout:
                orientation:"horizontal"
                size_hint:0.5,1
                BoxLayout:
                    orientation:"vertical"
                    size_hint:0.5,1
                    Label:
                        text:"Input Image"
                        valign:'middle'
                        halign:"left"
                        font_size:32
                    Label:
                        text:"Contrast Image"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:"Sengmented Image"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:"Classification"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:"Finish"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                BoxLayout:
                    orientation:"vertical"
                    size_hint:0.5,1
                    Label:
                        text:""
                        font_size:32
                        id:image1
                        valign:'middle'
                    Label:
                        text:""
                        font_size:32
                        id:image2
                        valign:'middle'
                    Label:
                        text:""
                        font_size:32
                        id:image3
                        valign:'middle'
                    Label:
                        text:""
                        font_size:32
                        id:image4
                        valign:'middle'
                    Label:
                        text:""
                        font_size:32
                        id:image5
                        valign:'middle'

            BoxLayout:
                orientation:"horizontal"
                size_hint:0.5,1
                BoxLayout:
                    orientation:"vertical"
                    size_hint:0.5,1
                    Label:
                        text:"Server IP"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:"Server Port"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:"K value"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:"Client IP"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:"Count"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                BoxLayout:
                    orientation:"vertical"
                    size_hint:0.5,1
                    Label:
                        text:""
                        halign:"left"
                        valign:'middle'
                        id:server_ip
                        font_size:32
                    Label:
                        text:""
                        halign:"left"
                        valign:'middle'
                        id:server_port
                        font_size:32
                    Label:
                        text:"5"
                        halign:"left"
                        valign:'middle'
                        font_size:32
                    Label:
                        text:""
                        halign:"left"
                        valign:'middle'
                        id:client_ip
                        font_size:32
                    Label:
                        text:"1"
                        halign:"left"
                        valign:'middle'
                        id:count
                        font_size:32
                
        BoxLayout:
            orientation:"horizontal"
            size_hint:1,0.1
            Button:
                text:"Stop Server"
                on_press:root.stop_server()
                font_size:32
            Button:
                text:"Start Server"
                on_press:root.start_server()
                font_size:32
''')




#other module
def getNetworkIp():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.connect(('<broadcast>', 0));ip_addr=s.getsockname()[0];s.close()
    return(ip_addr)

def resizeImage(imageName):
    basewidth = 100
    baseheight=100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,baseheight), Image.ANTIALIAS)
    img.save(imageName)

def getPredictedClass():
    # Predict
    image = cv2.imread('test_image{}.JPG'.format(count))
    #ray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([image.reshape(100, 100, 3)])
    #print(prediction)
    return numpy.argmax(prediction), (numpy.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] + prediction[0][5] + prediction[0][6] + prediction[0][7]))

def cnn_answer(predictedClass, confidence):
    textImage = numpy.zeros((300,512,3), numpy.uint8)
    className = ""
    if predictedClass == 0:
        className = "Apple scab"
        ans1=className
    elif predictedClass == 1:
        className = "Bacterial Spot"
        ans1=className
    elif predictedClass == 2:
        className = "Black Rot"
        ans1=className
    elif predictedClass == 3:
        className = "Cedar Apple Rust"
        ans1=className
    elif predictedClass == 4:
        className = "Early Blight"
        ans1=className
    elif predictedClass == 5:
        className = "Late Blight"
        ans1=className
    elif predictedClass == 6:
        className = "Mosaic Virus"
        ans1=className
    elif predictedClass == 7:
        className = "Powdery Mildew"
        ans1=className
    ans2=str(confidence * 100)
    return(ans1,ans2)


#functional class
class main_screen(Screen):
    def start_server(self):
        port = 60000                    # Reserve a port for your service.
        server_disp = self.ids["server_ip"]
        port_disp = self.ids["server_port"]
        port_disp.text=str(port)
        self.s = socket.socket()             # Create a socket object
        host = str(getNetworkIp())
        server_disp.text=host
        self.s.bind((host, port))            # Bind to the port
        self.s.listen(5)                     # Now wait for client connection.
        self.a=threading.Thread(target=self.server,args=(self,))
        self.a.start();

    def stop_server(self):
        try:
            self.s.close()
            server_disp = self.ids["server_ip"]
            port_disp = self.ids["server_port"]
            port_disp.text=""
            server_disp.text=""
        except:
            n=1
        #App.get_running_app().run()

        
    def server(self,*args):
        client_disp = self.ids["client_ip"]
        count_disp = self.ids["count"]
        image1=self.ids["image1"]
        image2=self.ids["image2"]
        image3=self.ids["image3"]
        image4=self.ids["image4"]
        image5=self.ids["image5"]
        try:
            while(True):
                global count
                count_disp.text="{}".format(count)
                count+=1
                conn, addr = self.s.accept()     # Establish connection with client.
                #print('Got connection from', addr)
                client_disp.text=addr[0]
                f=open("test_image{}.JPG".format(count),"wb");final_data=b''
                while(True):
                    data = conn.recv(1024)
                    if(not data or 'END' in str(data)):
                        final_data+=data[:-3]      
                        break
                    final_data+=data
                f.write(final_data);
                f.close()
                image1.text="done"
                conn.send("ok".encode())
                bgr = cv2.imread("test_image{}.JPG".format(count))
                lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
                lab_planes = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0)
                lab_planes[0] = clahe.apply(lab_planes[0])
                lab = cv2.merge(lab_planes)
                bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                cv2.imwrite("Contrast{}.JPG".format(count),bgr)
                f = open("Contrast{}.JPG".format(count),'rb')
                while(True):
                    l=f.read(1024)
                    if(not l):
                        conn.send("END".encode())
                        break;
                    conn.send(l)
                f.close()
                conn.recv(1024).decode()
                image2.text="done"
                image = cv2.imread("Contrast{}.JPG".format(count))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pixel_values = image.reshape((-1, 3))
                pixel_values = numpy.float32(pixel_values)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                k = 5
                _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                centers = numpy.uint8(centers)
            # flatten the labels array
                labels = labels.flatten()
            # convert all pixels to the color of the centroids
                segmented_image = centers[labels.flatten()]
            # reshape back to the original image dimension
                segmented_image = segmented_image.reshape(image.shape)
            # show the image
                #print("segmented image produced")
                cv2.imwrite("Segmented{}.JPG".format(count),segmented_image)
                f = open("Segmented{}.JPG".format(count),'rb')
                while(True):
                    l=f.read(1024)
                    if(not l):
                        conn.send("END".encode())
                        break;
                    conn.send(l)
                f.close()
                conn.recv(1024).decode()
                image3.text="done"
                img = io.imread("Segmented{}.JPG".format(count))
                gray = color.rgb2gray(img)
                image = img_as_ubyte(gray)

                bins = numpy.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) #16-bit
                inds = numpy.digitize(image, bins)

                max_value = inds.max()+1
                matrix = greycomatrix(inds, [1], [0, numpy.pi/4, numpy.pi/2, 3*numpy.pi/4], levels=max_value, normed=False, symmetric=False)


                gs = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                red_channel = img[:,:,0]
                green_channel = img[:,:,1]
                blue_channel = img[:,:,2]
                blue_channel[blue_channel == 255] = 0
                green_channel[green_channel == 255] = 0
                red_channel[red_channel == 255] = 0
                red_mean = numpy.mean(red_channel)
                green_mean = numpy.mean(green_channel)
                blue_mean = numpy.mean(blue_channel)
                red_var = numpy.std(red_channel)
                blue_var = numpy.std(blue_channel)
                green_var = numpy.std(green_channel)
                ContrastStats = numpy.mean(greycoprops(matrix, 'contrast'))
                CorrelationStats = numpy.mean(greycoprops(matrix, 'correlation'))
                HomogeneityStats = numpy.mean(greycoprops(matrix, 'homogeneity'))
                EnergyStats = numpy.mean(greycoprops(matrix, 'energy'))
                ASMStats = numpy.mean(greycoprops(matrix, 'ASM'))
                answer=[[red_mean,green_mean,blue_mean,red_var,blue_var,green_var,ContrastStats,CorrelationStats,HomogeneityStats,EnergyStats,ASMStats]]
                X=numpy.array(answer)
                ans=clf.predict(X)
                image4.source="done"
                resizeImage('test_image{}.JPG'.format(count))
                predictedClass, confidence = getPredictedClass();
                ans1,ans2=cnn_answer(predictedClass, confidence)
                conn.send("{}\n{}\n{}\n".format(output_array[ans[0]],ans1,ans2).encode())
                #print("{}\n{}\n{}\n".format(output_array[ans[0]],ans1,ans2))
                conn.recv(1024).decode()
                os.remove("test_image{}.JPG".format(count));
                os.remove("Contrast{}.JPG".format(count))
                os.remove("Segmented{}.JPG".format(count))
                image5.text="done"
                client_disp.text=" "
                image1.text=""
                image2.text=""
                image3.text=""
                image4.text=""
                image5.text=""

        except:
            image1.text=""
            image2.text=""
            image3.text=""
            image4.text=""
            image5.text=""
            client_disp.text=" "
            


#main class
class server_app(App):
    def build(self):
        return(sm)

    
#Basic or Root Widget i.e, ScreenManager
sm=ScreenManager()
sm.add_widget(main_screen(name="main_screen"))


#app call
if(__name__=="__main__"):
    server_app().run()
