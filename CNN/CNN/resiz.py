from PIL import Image
import os
import subprocess
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

#for i in range(0, :):
    # Mention the directory in which you wanna resize the images followed by the image name
    # resizeImage(r"Dataset/FingerImages/finger_" + str(i) + '.png')
subprocess.call("mkdir output",shell=True)
cwd=os.getcwd()
os.chdir(cwd+r'\Images');
a=[i for i in os.listdir() if ".jpg" or ".jpeg" or ".png" in i]
for i in a:
    resizeImage(i)


