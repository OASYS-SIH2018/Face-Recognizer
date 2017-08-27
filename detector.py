import cv2
import numpy as np
import sqlite3

faceDetect =cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
recog=cv2.face.createLBPHFaceRecognizer_create();
recog.load('recognizer\\trainingData.yml') #created from trainer.py
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_PLAIN,2,1,0,2)

def getData(Id):
    con=sqlite3.connect("Database1.db")
    cmd="select *from personaldata where ID="+str(Id)
    cursor=con.execute(cmd)
    data=None
    for row in cursor:
        data=row
    con.close()
    return data


while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);

    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recog.predict(gray[y:y+h,x:x+w])
        data=getData(id)
        if(data!=None): 
            cv2.cv.PutText(cv2.cv.fromarray(img),str(data[1]),(x,y+h+30),font,255);
            #cv2.cv.PutText(cv2.cv.fromarray(img),str(data[2]),(x,y+h+60),font,255);
           # cv2.cv.PutText(cv2.cv.fromarray(img),str(data[3]),(x,y+h+90),font,255);
        
    cv2.imshow("FACE",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()

        
