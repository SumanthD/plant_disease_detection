#all imports
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock 
import socket
import os
import sqlite3
from PIL import Image
#setting current working directory as final_year_project or creating final_year_project directory and making it as current working directory
'''
try:
    os.chdir("/sdcard/final_year_project")
except:
    os.chdir("/sdcard")
    os.makdir("final_year_project")
    os.chdir("/sdcard/final_year_project")
'''
#creating database to create memory or check what is there in previously created database
database=sqlite3.connect("final_year_database.db")
cursor_for_database=database.cursor()
cursor_for_database.execute("""CREATE TABLE IF NOT EXISTS ip_data(IP TEXT,port_number REAL)""")
cursor_for_database.execute("SELECT * FROM ip_data")
data_server_ip=[i for i in cursor_for_database];
cursor_for_database.close();database.close()
photo_number=0
#defining ip address of server from database
if(len(data_server_ip)==0):
    ip_addr_server="0"
    port_server="0"
else:
    ip_addr_server=data_server_ip[0][0];
    port_server=data_server_ip[0][1]
pic_from_sdcard=""


#GUI using Kivy Language
Builder.load_string('''
<CameraClick>:
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation:'horizontal'
            size_hint:1,0.09
            Button:
                text:"Settings"
                id:setting_button
                size_hint:0.2,1
                on_press:root.setting_tab()
                font_size:32
            Button:
                id:test_button
                text: 'test'
                font_size:32
                size_hint:0.2,1
                on_press: root.test()

        Widget:
            size_hint:1,1
            id:camera_box

            Camera:
                id: camera
                resolution: (1280,720)
                keep_ratio:True
                size_hint:1,0.9
                size: camera_box.height, camera_box.width
                center: self.size and camera_box.center
                play: True
                allow_stretch: True
                canvas.before:
                    PushMatrix
                    Rotate:
                        angle: -90
                        origin: self.center
                canvas.after:
                    PopMatrix
        BoxLayout:
            orientation:"horizontal"
            size_hint:1,0.1
            Button:
                id:capture_button
                text: 'Capture'
                font_size:32
                on_press: root.capture()



<analysis_answer>:
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            size_hint:1,0.1
            orientation: 'horizontal'
            Button:
                text:"Contrast Enhanced"
                font_size:32
                pos_hint_y:0
                on_press:root.contrast_enhancement_image()
            Button:
                text:"Segmented"
                font_size:32
                pos_hint_y:0
                on_press:root.contrast_segmented_image()


        Image:
            id: image
            size_hint:1,0.7
            source:"final_year_icon.gif"
            allow_stretch: True

        BoxLayout:
            orientation:'horizontal'
            size_hint:1,0.1
            BoxLayout:
                orientation:'vertical'                
                Label:
                    text:"SVM"
                Label:
                    text:"CNN"
            BoxLayout:
                orientation:'vertical'                
                Label:
                    text:""
                    id: svm_answer
                Label:
                    text:""
                    id: cnn_answer
            BoxLayout:
                orientation:'vertical'                
                Label:
                    text:"nil"
                Label:
                    id: accuracy
                    text:""
                    

        Button:
            text: 'Clear!'
            font_size:32
            size_hint:1,0.1
            on_press:root.clear_everything()

<setting>:
    BoxLayout:
        orientation:"vertical"
        BoxLayout:
            orientation:'vertical'
            hint_size:1,0.1
            BoxLayout:
                orientation:'horizontal'
                Label:
                    text:'Server IP'
                    font_size:40
                TextInput:
                    id:server_ip
                    text:root.ip_addr_server
                    multiline:False
                    font_size:40
            BoxLayout:
                orientation:'horizontal'
                Label:
                    text:'Server Port'
                    font_size:40
                TextInput:
                    id:server_port
                    text:root.port_server
                    multiline:False
                    font_size:40
        BoxLayout:
            orientation:"vertical"
            hint_size:1,0.85
            Label:
                text:""
        BoxLayout:
            orientation:'horizontal'
            hint_size:1,0.05
            Button:
                text:'Back'
                id:from_setting_to_capture
                font_size:32
                on_press:root.back()
            Button:
                text:'Save'
                id:save_setting
                font_size:32
                on_press:root.save()


<send_file>:
    BoxLayout:
        orientation:"vertical"
        TextInput:
            id:input_file
            size_hint:1,0.1
            multiline:False
            bold:True
            font_size:40
        Label:
            text:""
            size_hint:1,0.7
        Button:
            text:'Send'
            id:send_picture
            font_size:32
            size_hint:1,0.1
            on_press:root.test()
        Button:
            text:"Back"
            id:back_to_capture
            font_size:32
            size_hint:1,0.1
            on_press:root.back_to_capture()
            
<server_error>:
    BoxLayout:
        orientation:'vertical'
        Label:
            text:"Server not found"
            font_size:50
            size_hint:1,0.9
        Button:
            text:"Back!"
            on_press:root.back()
            size_hint:1,0.1
            font_size:32

<file_error>:
    BoxLayout:
        orientation:'vertical'
        Label:
            text:"File not found"
            font_size:50
            size_hint:1,0.9
        Button:
            text:"Back!"
            on_press:root.back()
            size_hint:1,0.1
            font_size:32
               
''')

#3 screens definintion using class

#CameraClick screen
class CameraClick(Screen):
    def capture(self):
        camera = self.ids['camera']
        capture_button= self.ids['capture_button']
        setting_button= self.ids['setting_button']
        test_button= self.ids['test_button']
        camera.export_to_png("{}/1.png".format(os.getcwd()))
        img=Image.open("1.png")
        img.convert("RGB").save("1.JPG");os.remove("1.png")
        n=self.send_to_server("1.JPG")
        if(n==1):
            self.get_contrast_image()
            self.get_segmented_image()
            self.answer_file()
        else:
            os.remove("1.JPG");
            sm.current="server_error"

    def setting_tab(self):
        sm.current="setting"

    def send_to_server(self,file_to_send):
        try:
            self.s = socket.socket()             # Create a socket object
            host = ip_addr_server           # Get local machine name
            port = port_server              # Reserve a port for your service.
            self.s.connect((str(host), int(port)))
            f = open(file_to_send,'rb')
            while(True):
                l=f.read(1024)
                if(not l):
                    self.s.send("END".encode())
                    break;
                self.s.send(l)
            f.close()
            self.s.recv(1024).decode()
            return(1)
        except:
            return(0)
        
    
    def get_contrast_image(self):
        f=open("test_image_contrast_{}.JPG".format(photo_number),"wb");final_data=b''
        while(True):
            data = self.s.recv(1024)
            if(not data or 'END' in str(data)):
                final_data+=data
                break
            final_data+=data
        f.write(final_data)
        f.close()
        self.s.send("Ok received contrast image".encode())
        

    def get_segmented_image(self):
        f=open("test_image_segmented_{}.JPG".format(photo_number),"wb");final_data=b''
        while(True):
            data = self.s.recv(1024)
            if(not data or 'END' in str(data)):
                final_data+=data
                break
            final_data+=data
        f.write(final_data)
        f.close()
        self.s.send("Ok received segmented image".encode())
        


    def test(self):
        sm.current="send_file"
    
    def answer_file(self):
        data=self.s.recv(1024)
        self.s.send("ok received answer file".encode())
        f=open("answer_{}.txt".format(photo_number),"wb");
        f.write(data);
        f.close()
        self.s.close()
        sm.current="analysis"

   
#Answer screen
class analysis_answer(Screen):
    def contrast_segmented_image(self):
        image_display = self.ids['image']
        image_display.source="test_image_segmented_{}.JPG".format(photo_number);

    def contrast_enhancement_image(self):
        image_display = self.ids['image']
        image_display.source="test_image_contrast_{}.JPG".format(photo_number);
        svm_answer = self.ids['svm_answer']
        cnn_answer = self.ids['cnn_answer']
        accuracy_answer= self.ids['accuracy']
        f=open("answer_{}.txt".format(photo_number),"r");
        ans=f.read();ans=ans.split("\n");
        svm_answer.text=ans[0]
        cnn_answer.text=ans[1]
        accuracy_answer.text=str(round(float(ans[2]),2))
        
    def clear_everything(self):
        global photo_number
        image_display = self.ids['image'];
        image_display.source="final_year_icon.gif"
        svm_answer = self.ids['svm_answer']
        cnn_answer = self.ids['cnn_answer']
        accuracy_answer= self.ids['accuracy']
        os.remove("test_image_segmented_{}.JPG".format(photo_number));
        try:
            os.remove("1.JPG");
        except:
            n=1
        os.remove("test_image_contrast_{}.JPG".format(photo_number));
        os.remove("answer_{}.txt".format(photo_number))
        svm_answer.text=""
        cnn_answer.text=""
        accuracy_answer.text=""
        photo_number+=1
        sm.current="capture"




#setting screen
class setting(Screen):
    ip_addr_server=str(ip_addr_server)
    port_server=str(int(port_server))

    def back(self):
        sm.current="capture"

    def save(self):
        database=sqlite3.connect("final_year_database.db")
        cursor_for_database=database.cursor()
        ip_addr_server=str(self.ids['server_ip'].text)
        port_server=str(self.ids['server_port'].text);
        cursor_for_database.execute("""CREATE TABLE IF NOT EXISTS ip_data(IP TEXT,port_number REAL)""")
        cursor_for_database.execute("DROP TABLE IF EXISTS ip_data")
        cursor_for_database.execute("CREATE TABLE IF NOT EXISTS ip_data(IP TEXT,port_number REAL)")
        cursor_for_database.execute("""INSERT INTO ip_data(IP,port_number) VALUES (?,?)""",(ip_addr_server,float(port_server)));
        database.commit();cursor_for_database.close();database.close();
        port_server=int(port_server)
        sm.current="capture"

   
class send_file(Screen):
    def test(self):
        global pic_from_sdcard;
        pic_from_sdcard="{}.JPG".format(self.ids['input_file'].text)
        try:
            f=open(pic_from_sdcard,"rb");
        except:
            sm.current="file_error"
            return(0)
        n=CameraClick.send_to_server(CameraClick,pic_from_sdcard)
        if(n==1):
            CameraClick.get_contrast_image(CameraClick)
            CameraClick.get_segmented_image(CameraClick)
            CameraClick.answer_file(CameraClick)
            self.ids['input_file'].text=""
        else:
            sm.current="server_error"
    def back_to_capture(self):
        sm.current="capture"
        
class server_error(Screen):
    def back(self):
        sm.current="capture"

class file_error(Screen):
    def back(self):
        sm.current="send_file"
#Basic or Root Widget i.e, ScreenManager
sm=ScreenManager()
sm.add_widget(CameraClick(name="capture"))
sm.add_widget(analysis_answer(name='analysis'))
sm.add_widget(setting(name='setting'))
sm.add_widget(send_file(name='send_file'))
sm.add_widget(server_error(name='server_error'))
sm.add_widget(file_error(name='file_error'))


#main class
class TestCamera(App):
    def build(self):
        return(sm)
	

if(__name__=="__main__"):
    TestCamera().run()
