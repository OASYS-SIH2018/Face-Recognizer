import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.createLBPHFaceRecognizer();
path='Dataset'

def getImagesId(path):
    imgPaths=[os.path.join(path,f) for f in os.listdir(path)]
    #takes all images in the dir specified by the path and stores it in f and the joins them with the path including \
    print imgPaths
    #getImagesId(path);
    faces=[]
    IDs=[]
    for imgPath in imgPaths:
        faceImg=Image.open(imgPath).convert('L');#convert into grayscale
        faceNp=np.array(faceImg,'uint8')#create numpyarr of type unsignedint opencv works with np arr only
        ID=int(os.path.split(imgPath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        #cv2.imshow("Training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs),faces
ID,faces=getImagesId(path)
recognizer.train(faces,ID)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()

        
