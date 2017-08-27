import cv2
import numpy as np
import sqlite3

faceDetect =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)# (0) -> In-built webcam

#Input     
ID=raw_input('User id:')
Name=raw_input('Name:')
insert(ID,Name)
Num=0

def insert(Id,Name):
    #establish connection
    con=sqlite3.connect("Database1.db")

    #check if existing or new user 
    cmd="SELECT * FROM PersonalData WHERE ID="+str(Id)
    cursor=con.execute(cmd)
    recordExist=0
    for row in cursor:
        recordExist=1
    #Existing user -> update his new Name
    if(recordExist==1):
        cmd="update personaldata set name="+str(Name)+"where ID="+str(Id)
    else:
    #New user -> add new Name and Id in the database
        cmd="insert into personaldata(ID,Name) values("+str(Id)+","+str(Name)+")"
    cursor1=con.execute(cmd)
    con.commit()
    con.close()

#Capture 40 snapshots of user to create Train_data
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        Num=Num+1
        cv2.imwrite("Dataset/User."+str(ID)+"."+str(Num)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow("FACE",img)
    cv2.waitKey(100);
    if(Num>40):
        break;
    
cam.release()
cv2.destroyAllWindows()

        
