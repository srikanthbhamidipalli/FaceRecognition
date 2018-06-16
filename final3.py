#script for detecting live faces,auto cropping and saving.. 
import cv2,os,time
from skimage.measure import compare_ssim as ssim #importing required modules

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#importing haar cascade frontal face detector
cap = cv2.VideoCapture(r"C:\Users\SRIKANTH\Desktop\1.mp4")#capturing a video
count=0#to give the numbering for captured faces
list1=[]#to store all the captured faces
list2=[]#to store the similar faces
print(time.ctime())#printing the current time
while True:
    ret, frame = cap.read()#reading video frames into frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#for changing colorspaces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=7,minSize=(30,30),
                                          flags=cv2.CASCADE_SCALE_IMAGE
                                          )#to find faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)#drawing a green coloured rectangle of thickness 2px
        sub_face = frame[y:y+h, x:x+h]
        
 
        # perform the actual resizing of the image (100*100) and show it
        resized = cv2.resize(sub_face, (100,100), interpolation = cv2.INTER_LINEAR)#for zooming
        FaceFileName = "C:/Python35/faces11/face{:d}.jpg".format(count)#destiny for saving faces
        list1.append(FaceFileName)#appending all the faces in a list
        count+=1
        cv2.imwrite(FaceFileName, resized)#saving captured faces
        
    cv2.imshow('Video', frame)#playing the video in new window

    if cv2.waitKey(1) & 0xFF == ord('q'):#for quitting
        break
    
print(time.ctime())
print("face count is",count)
cap.release()
cv2.destroyAllWindows()
for i in range(count):
    for j in range(i+1,count):
        a=list1[i]
        b=list1[j]
        c=cv2.imread(a)#reading an image
        d=cv2.imread(b)
        c=cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)#for changing colorspaces
        d=cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        s=ssim(c,d)#structural similarity index for pixel by pixel comparision
        if s>0.5:
            list2.append(b)#appending similar faces
for k in range(len(list2)+1):
    try:
        os.remove(list2[k])#removing similar faces
    except FileNotFoundError:
        continue
    except IndexError:
        print("Done")
